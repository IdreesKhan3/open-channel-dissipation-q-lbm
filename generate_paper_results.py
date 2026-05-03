#!/usr/bin/env python3
"""Generate manuscript figures from Taylor--Green D3Q19 runs.

Outputs theorem_error_dm_only.pdf (max absolute moment error vs time, baseline_jax
audit) and tgv_ke_dissipation.pdf (mean kinetic energy and volume-mean viscous
dissipation, ref_mrt vs baseline_jax), aligned with the main text defaults.

Author: Muhammad Idrees Khan
Version: 1.0.0
"""
from __future__ import annotations

__version__ = "1.0.0"

import argparse
import os
import sys
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib._mathtext as _mathtext
import numpy as np

from baseline_jax import run_baseline_open_channel_jax
from ref_mrt import run_ref_mrt

_FIG_W_IN = 6.35  # theorem_error_dm_only.pdf only (TGV two-panel width is _FIG_W_KE_SIDE)
_FIG_H_THEOREM = 6.8 * 0.65
_FIG_W_KE_SIDE = 6.8 * 1.48
_FIG_H_KE_SIDE = 6.8 * 0.65
TGV_X_MAX = 20.0
# PDF typography (pt).
_FIG_FONT_PT = 13
_FIG_LABEL_PT = 13
_FIG_TICK_PT = 12
_FIG_LEGEND_PT = 12
# Axis frame and ticks (spine width; major tick mark thickness).
_FIG_AXES_LINEWIDTH = 1.35
_FIG_TICK_MAJOR_WIDTH = 1.05
# Colorblind-friendly pair: blue (IBM palette) + red distinct from warm orange.
_TGV_COLOR_CLASSICAL = "#0072B2"
_TGV_COLOR_OPEN = "#B83232"
_TGV_LINEWIDTH = 2.25

_TGV_CPTP_MARKER = "s"
_TGV_CPTP_MARKERSIZE = 6.5
_TGV_CPTP_MARKER_EDGE = "#FFFFFF"
_TGV_CPTP_MARKER_EDGEWIDTH = 1.0
_TGV_CPTP_TARGET_MARKERS = 48


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--timesteps", type=int, default=2000)
    ap.add_argument("--u0", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.5035)
    ap.add_argument(
        "-o",
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: directory of this script)",
    )
    args = ap.parse_args()

    outdir = args.outdir.resolve() if args.outdir else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    out_pdf = outdir / "theorem_error_dm_only.pdf"
    out_ke_pdf = outdir / "tgv_ke_dissipation.pdf"
    debug_history: list = []
    time_open, ke_open, eps_open = run_baseline_open_channel_jax(
        args.nx,
        args.nx,
        args.nx,
        args.u0,
        args.tau,
        args.timesteps,
        quiet=True,
        debug_history=debug_history,
        collect_strain_dissipation=True,
    )
    time_ref, ke_ref, eps_ref = run_ref_mrt(
        args.nx,
        args.nx,
        args.nx,
        args.u0,
        args.tau,
        args.timesteps,
        quiet=True,
        collect_strain_dissipation=True,
    )
    time_ref = np.asarray(time_ref, dtype=np.float64)
    time_open = np.asarray(time_open, dtype=np.float64)
    ke_ref = np.asarray(ke_ref, dtype=np.float64)
    ke_open = np.asarray(ke_open, dtype=np.float64)
    eps_ref = np.asarray(eps_ref, dtype=np.float64)
    eps_open = np.asarray(eps_open, dtype=np.float64)
    if time_ref.shape != time_open.shape or not np.allclose(time_ref, time_open):
        raise RuntimeError("ref_mrt and baseline time axes differ; check shared nx,u0,timesteps.")
    time_ke = time_ref
    t_end = float(time_ke[-1])
    if t_end <= 0.0:
        raise RuntimeError("TGV time series has non-positive duration.")
    x_tgv = (time_ke / t_end) * TGV_X_MAX
    cptp_markevery = max(
        1, int(np.ceil(x_tgv.size / float(_TGV_CPTP_TARGET_MARKERS)))
    )
    if not debug_history:
        raise RuntimeError("debug_history is empty.")

    n_dbg = len(debug_history)
    time_axis = np.asarray(time_open, dtype=np.float64)
    if time_axis.size != n_dbg:
        dt = 2.0 * np.pi / float(args.nx) * float(args.u0)
        time_axis = (np.array([d["step"] for d in debug_history], dtype=np.float64) - 1.0) * dt

    plt.rcParams.update({
        "font.size": _FIG_FONT_PT,
        "axes.labelsize": _FIG_LABEL_PT,
        "xtick.labelsize": _FIG_TICK_PT,
        "ytick.labelsize": _FIG_TICK_PT,
        "legend.fontsize": _FIG_LEGEND_PT,
        "axes.linewidth": _FIG_AXES_LINEWIDTH,
        "xtick.major.width": _FIG_TICK_MAJOR_WIDTH,
        "ytick.major.width": _FIG_TICK_MAJOR_WIDTH,
        "legend.frameon": False,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "mathtext.default": "regular",
    })
    old_shrink = _mathtext.SHRINK_FACTOR
    
    _mathtext.SHRINK_FACTOR = 1.0

    fig, ax = plt.subplots(figsize=(_FIG_W_IN, _FIG_H_THEOREM))
    ax.semilogy(
        time_axis,
        [d["err_dm_max"] for d in debug_history],
        color="#0072B2",
        label=r"$\mathrm{max}|\delta m'-\delta m_{\mathrm{target}}|$",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Error")
    ax.legend(loc="best")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))
    ax.xaxis.grid(True, which="major", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=600)
    plt.close(fig)

    fig2, (ax_ke, ax_eps) = plt.subplots(
        1, 2, figsize=(_FIG_W_KE_SIDE, _FIG_H_KE_SIDE), sharex=True
    )
    ax_ke.semilogy(
        x_tgv,
        ke_ref,
        color=_TGV_COLOR_CLASSICAL,
        linestyle="-",
        label="Classical MRT",
        linewidth=_TGV_LINEWIDTH,
        zorder=1,
    )
    ax_ke.semilogy(
        x_tgv,
        ke_open,
        color=_TGV_COLOR_OPEN,
        linestyle="--",
        label="CPTP MRT",
        linewidth=_TGV_LINEWIDTH,
        zorder=2,
        marker=_TGV_CPTP_MARKER,
        markersize=_TGV_CPTP_MARKERSIZE,
        markevery=cptp_markevery,
        markerfacecolor=_TGV_COLOR_OPEN,
        markeredgecolor=_TGV_CPTP_MARKER_EDGE,
        markeredgewidth=_TGV_CPTP_MARKER_EDGEWIDTH,
    )
    ax_ke.set_ylabel(r"$E_k = \frac{1}{2}\langle u_i u_i \rangle$")
    ax_ke.set_xlabel("Time")
    ax_ke.set_xlim(0.0, TGV_X_MAX)
    ax_ke.legend(loc="best")
    ax_ke.xaxis.grid(True, which="major", alpha=0.15)
    ax_ke.yaxis.grid(True, which="major", alpha=0.15)

    ax_eps.plot(
        x_tgv,
        eps_ref,
        color=_TGV_COLOR_CLASSICAL,
        linestyle="-",
        linewidth=_TGV_LINEWIDTH,
        label="Classical MRT",
        zorder=1,
    )
    ax_eps.plot(
        x_tgv,
        eps_open,
        color=_TGV_COLOR_OPEN,
        linestyle="--",
        linewidth=_TGV_LINEWIDTH,
        label="CPTP MRT",
        zorder=2,
        marker=_TGV_CPTP_MARKER,
        markersize=_TGV_CPTP_MARKERSIZE,
        markevery=cptp_markevery,
        markerfacecolor=_TGV_COLOR_OPEN,
        markeredgecolor=_TGV_CPTP_MARKER_EDGE,
        markeredgewidth=_TGV_CPTP_MARKER_EDGEWIDTH,
    )
    ax_eps.legend(loc="best")
    ax_eps.set_xlabel("Time")
    ax_eps.set_ylabel(r"$\varepsilon = 2\nu_0 \langle S_{ij} S_{ij} \rangle$")
    ax_eps.set_xlim(0.0, TGV_X_MAX)
    ax_eps.xaxis.set_major_locator(mticker.MaxNLocator(nbins=6, integer=False))
    ax_eps.xaxis.grid(True, which="major", alpha=0.15)
    ax_eps.yaxis.grid(True, which="major", alpha=0.15)
    fig2.align_ylabels([ax_ke, ax_eps])
    fig2.tight_layout()
    fig2.savefig(out_ke_pdf, dpi=600)
    plt.close(fig2)
    _mathtext.SHRINK_FACTOR = old_shrink
    return 0


if __name__ == "__main__":
    sys.exit(main())
