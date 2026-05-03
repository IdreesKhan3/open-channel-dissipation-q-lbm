#!/usr/bin/env python3
"""Theorem-audit table for D3Q19 Taylor--Green (same pipeline as baseline_jax).

Trajectory rows: running max of err_dm_max from debug_history over all steps
(same diagnostic as theorem_error_dm_only.pdf). Endpoint rows: Kraus map at
lambda in {-1, 0, +1} on a real TGV dm snapshot.

Writes theorem_audit_table.json next to this script. Run from this directory:
``python3 run_theorem_audit.py`` (optional: ``JAX_PLATFORMS=cuda`` and flags such
as ``--nx 32 --timesteps 500``).

Endpoint lambdas {-1, 0, +1}: the snapshot dm is run through Kraus tensors
built for those lambda values only, temporarily ignoring the MRT relaxation rates
that come from tau in the main driver (regression sanity check on real-flow dm).

Decode should match lambda * dm to roundoff at these extremes (ideal damping amplitude
factor 0 or 1).

Author: Muhammad Idrees Khan
Version: 1.0.0
"""
from __future__ import annotations

__version__ = "1.0.0"

import json
import os
import sys
import time
from pathlib import Path

if "JAX_PLATFORMS" not in os.environ:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from baseline_jax import (  # noqa: E402
    DISSIPATIVE,
    NODE_VELOCITIES,
    SCALE_TOL,
    _apply_kraus_jax,
    _build_kraus,
    build_M_matrix_D3Q19,
    get_equilibrium,
    rho0,
    run_baseline_open_channel_jax,
)


def _endpoint_error_on_snapshot(dm_np, lam_val):
    """Apply the two-rail channel with lambda_r = lam_val on every dissipative
    mode to a real dm snapshot, return max |dm' - lam * dm| over sites and
    dissipative modes."""
    alpha = abs(lam_val)
    need_swap = lam_val < 0.0
    kraus = _build_kraus(alpha, need_swap)
    worst = 0.0
    for r in DISSIPATIVE:
        dm_r = jnp.asarray(dm_np[..., r])
        S_r = jnp.maximum(jnp.abs(dm_r), SCALE_TOL)
        dm_prime_r = _apply_kraus_jax(dm_r, S_r, kraus)
        err = float(jnp.max(jnp.abs(dm_prime_r - lam_val * dm_r)))
        if err > worst:
            worst = err
    return worst


def _grab_tgv_dm_snapshot(nx: int, u0: float) -> np.ndarray:
    """Same ``dm = m - m_eq`` as the first collision inside ``baseline_jax.update_local``.

    Uses the TGV equilibrium populations from ``get_equilibrium``, then recomputes
    ``m``, ``m_eq``, and ``u`` exactly as ``run_baseline_open_channel_jax`` does before
    applying Kraus. Initialization uses equilibrium populations from ``u0`` and ``nx``
    only; ``tau`` and the derived MRT relaxation vector ``s`` appear in collisions, not
    in constructing this ``dm`` snapshot.

    Returned array is NumPy CPU, shape ``(nx, nx, nx, 19)``.
    """
    dx = 2.0 * jnp.pi / nx
    x = (jnp.arange(nx) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    ux = u0 * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    uy = -u0 * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    uz = jnp.zeros_like(X)
    u_init = jnp.stack([ux, uy, uz], axis=-1)
    rho_init = jnp.ones((nx, nx, nx))
    dv_prev = get_equilibrium(u_init, rho_init)

    rho = jnp.sum(dv_prev, axis=-1)
    rho_safe = jnp.where(rho < 1e-15, 1.0, rho)
    rhoinv = 1.0 / rho_safe
    u = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, dv_prev) * rhoinv[..., None]
    jx = rho * u[..., 0]
    jy = rho * u[..., 1]
    jz = rho * u[..., 2]
    j_sq = jx**2 + jy**2 + jz**2
    delta_rho = rho - rho0

    M_ = build_M_matrix_D3Q19()
    m = jnp.einsum("ab,ijkb->ijka", M_, dv_prev)
    m_eq = jnp.zeros_like(m)
    m_eq = m_eq.at[..., 0].set(delta_rho)
    m_eq = m_eq.at[..., 1].set(-11.0 * delta_rho + 19.0 * j_sq / rho0)
    m_eq = m_eq.at[..., 2].set((-475.0 / 63.0) * j_sq / rho0)
    m_eq = m_eq.at[..., 3].set(jx)
    m_eq = m_eq.at[..., 4].set(-(2.0 / 3.0) * jx)
    m_eq = m_eq.at[..., 5].set(jy)
    m_eq = m_eq.at[..., 6].set(-(2.0 / 3.0) * jy)
    m_eq = m_eq.at[..., 7].set(jz)
    m_eq = m_eq.at[..., 8].set(-(2.0 / 3.0) * jz)
    m_eq = m_eq.at[..., 9].set((2.0 * jx**2 - jy**2 - jz**2) / rho0)
    m_eq = m_eq.at[..., 10].set(0.0)
    m_eq = m_eq.at[..., 11].set((jy**2 - jz**2) / rho0)
    m_eq = m_eq.at[..., 12].set(0.0)
    m_eq = m_eq.at[..., 13].set(jx * jy / rho0)
    m_eq = m_eq.at[..., 14].set(jy * jz / rho0)
    m_eq = m_eq.at[..., 15].set(jx * jz / rho0)
    m_eq = m_eq.at[..., 16].set(0.0)
    m_eq = m_eq.at[..., 17].set(0.0)
    m_eq = m_eq.at[..., 18].set(0.0)
    return np.asarray(jax.device_get(m - m_eq))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--nx", type=int, default=64)
    ap.add_argument("--tau", type=float, default=0.5035)
    ap.add_argument("--u0", type=float, default=0.1)
    ap.add_argument("--timesteps", type=int, default=2000)
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(
        f"Decaying TGV audit: nx={args.nx}^3, tau={args.tau}, "
        f"u0={args.u0}, timesteps={args.timesteps}"
    )

    t0 = time.time()
    debug_history = []
    _, _ = run_baseline_open_channel_jax(
        args.nx,
        args.nx,
        args.nx,
        args.u0,
        args.tau,
        timesteps=args.timesteps,
        quiet=True,
        debug_history=debug_history,
    )
    t1 = time.time()
    print(f"Run done in {t1 - t0:.1f}s; {len(debug_history)} steps recorded.")

    err_std = max(d["err_dm_max"] for d in debug_history)

    dm_snapshot = _grab_tgv_dm_snapshot(args.nx, args.u0)
    err_m1 = _endpoint_error_on_snapshot(dm_snapshot, -1.0)
    err_z = _endpoint_error_on_snapshot(dm_snapshot, 0.0)
    err_p1 = _endpoint_error_on_snapshot(dm_snapshot, +1.0)

    results = {
        "setup": {
            "nx": args.nx,
            "tau": args.tau,
            "u0": args.u0,
            "timesteps": args.timesteps,
            "scale_mode": "adaptive",
            "flow": "decaying TGV (no forcing)",
            "runtime_seconds": round(t1 - t0, 2),
        },
        "standard_lambda_tgv": float(err_std),
        "endpoint_lambda_-1_tgv": float(err_m1),
        "endpoint_lambda_0_tgv": float(err_z),
        "endpoint_lambda_+1_tgv": float(err_p1),
        "adaptive_S_r_tgv": float(err_std),
    }
    print()
    for k, v in results.items():
        if isinstance(v, float):
            print(f"{k:32s}  {v:.3e}")
    out = Path(__file__).resolve().parent / "theorem_audit_table.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
