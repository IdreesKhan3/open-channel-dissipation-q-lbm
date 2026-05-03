#!/usr/bin/env python3
"""Random (dm, lambda) stress tests for the Kraus two-rail map (uses baseline_jax helpers)."""

from __future__ import annotations

import argparse
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

try:
    from baseline_jax import (
        SCALE_TOL,
        _apply_kraus_jax,
        _build_kraus,
    )
except ImportError:
    try:
        from only_OpenChanel_BaseLine.baseline_jax import (
            SCALE_TOL,
            _apply_kraus_jax,
            _build_kraus,
        )
    except ImportError as exc:
        raise ImportError(
            "Could not import baseline_jax: place run_synthetic_audit.py in the "
            "same folder as baseline_jax.py, or expose only_OpenChanel_BaseLine "
            "on PYTHONPATH."
        ) from exc


def _apply_for_lambda(dm_vec_np, lam, scale_mode="adaptive", S_fixed=None):
    """Decoded channel output for one lambda (adaptive scale or fixed S_r >= |dm|)."""
    dm_r = jnp.asarray(dm_vec_np, dtype=jnp.float64)
    if scale_mode == "adaptive":
        S_r = jnp.maximum(jnp.abs(dm_r), SCALE_TOL)
    elif scale_mode == "fixed":
        if S_fixed is None:
            raise ValueError("S_fixed must be provided when scale_mode='fixed'.")
        S_val = float(S_fixed)
        max_abs_dm = float(jnp.max(jnp.abs(dm_r)))
        if S_val < max_abs_dm:
            raise ValueError(
                f"Invalid fixed scale: S_fixed={S_val:.3e} < max|dm|="
                f"{max_abs_dm:.3e}; Theorem 1 requires S_r >= |dm_r|."
            )
        S_r = jnp.full_like(dm_r, S_val)
    else:
        raise ValueError(f"unknown scale_mode {scale_mode!r}")
    kraus = _build_kraus(abs(lam), lam < 0.0)
    dm_prime = _apply_kraus_jax(dm_r, S_r, kraus)
    return np.asarray(jax.device_get(dm_prime))


def sweep_grid_lambda(n_lambda=101, n_dm=1000, X=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    lambdas = np.linspace(-1.0, 1.0, n_lambda)
    worst = 0.0
    for lam in lambdas:
        dm = rng.uniform(-X, X, size=n_dm)
        dm_p = _apply_for_lambda(dm, float(lam))
        err = float(np.max(np.abs(dm_p - lam * dm)))
        if err > worst:
            worst = err
    return worst, n_lambda * n_dm


def sweep_random_joint(n_lambda=200, n_dm=200, X=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(1)
    lambdas = rng.uniform(-1.0, 1.0, size=n_lambda)
    worst = 0.0
    for lam in lambdas:
        dm = rng.uniform(-X, X, size=n_dm)
        dm_p = _apply_for_lambda(dm, float(lam))
        err = float(np.max(np.abs(dm_p - lam * dm)))
        if err > worst:
            worst = err
    return worst, n_lambda * n_dm


def sweep_boundary(n_dm=10000, X=1.0, rng=None, eps=1e-12):
    if rng is None:
        rng = np.random.default_rng(2)
    edge_lams = [-1.0, -1.0 + eps, -eps, 0.0, +eps, 1.0 - eps, 1.0]
    edge_dm = np.array([-X, -X + eps, -eps, 0.0, +eps, X - eps, X],
                       dtype=float)
    worst = 0.0
    total = 0
    for lam in edge_lams:
        dm_rand = rng.uniform(-X, X, size=n_dm)
        dm = np.concatenate([dm_rand, edge_dm])
        dm_p = _apply_for_lambda(dm, float(lam))
        err = float(np.max(np.abs(dm_p - lam * dm)))
        if err > worst:
            worst = err
        total += dm.size
    return worst, total


def sweep_scale_sensitivity(n_samples=200, X=1.0, rng=None, n_S=50):
    if rng is None:
        rng = np.random.default_rng(3)
    worst = 0.0
    total = 0
    for _ in range(n_samples):
        dm = float(rng.uniform(-X, X))
        lam = float(rng.uniform(-1.0, 1.0))
        if abs(dm) < 1e-15:
            dm = 0.0
        S_min = max(abs(dm), float(SCALE_TOL))
        S_vals = np.geomspace(S_min, S_min * 1e6, num=n_S)
        for S in S_vals:
            dm_p = _apply_for_lambda([dm], lam, scale_mode="fixed",
                                     S_fixed=float(S))
            err = float(np.abs(dm_p[0] - lam * dm))
            if err > worst:
                worst = err
            total += 1
    return worst, total


def sweep_exact_corners(X=1.0):
    dms = np.array([-X, -X / 2.0, 0.0, X / 2.0, X], dtype=float)
    lams = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    worst = 0.0
    total = 0
    for lam in lams:
        dm_p = _apply_for_lambda(dms, float(lam))
        err = float(np.max(np.abs(dm_p - lam * dms)))
        if err > worst:
            worst = err
        total += dms.size
    return worst, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--X", type=float, default=1.0,
                    help="dm domain half-width; draws dm from [-X, X].")
    ap.add_argument("--n-lambda-grid", type=int, default=101)
    ap.add_argument("--n-dm-grid", type=int, default=1000)
    ap.add_argument("--n-lambda-rand", type=int, default=200)
    ap.add_argument("--n-dm-rand", type=int, default=200)
    ap.add_argument("--n-boundary", type=int, default=10000)
    ap.add_argument("--n-scale-samples", type=int, default=200)
    ap.add_argument("--n-scale-points", type=int, default=50)
    args = ap.parse_args()

    print(f"JAX backend: {jax.default_backend()}  devices: {jax.devices()}")
    print(f"Stencil-free synthetic audit; dm in [-{args.X}, {args.X}], lam in [-1, 1]")

    t0 = time.time()
    err_grid, n_grid = sweep_grid_lambda(
        args.n_lambda_grid, args.n_dm_grid, args.X,
        rng=np.random.default_rng(0))
    t1 = time.time()
    err_rand, n_rand = sweep_random_joint(
        args.n_lambda_rand, args.n_dm_rand, args.X,
        rng=np.random.default_rng(1))
    t2 = time.time()
    err_bdy, n_bdy = sweep_boundary(
        args.n_boundary, args.X,
        rng=np.random.default_rng(2))
    t3 = time.time()
    err_scale, n_scale = sweep_scale_sensitivity(
        args.n_scale_samples, args.X,
        rng=np.random.default_rng(3),
        n_S=args.n_scale_points)
    t4 = time.time()
    err_corners, n_corners = sweep_exact_corners(args.X)
    t5 = time.time()

    results = {
        "setup": {
            "X": args.X,
            "lambda_domain": [-1.0, 1.0],
            "dm_domain": [-args.X, args.X],
            "timings_sec": {
                "grid": round(t1 - t0, 3),
                "random_joint": round(t2 - t1, 3),
                "boundary": round(t3 - t2, 3),
                "scale_sensitivity": round(t4 - t3, 3),
                "exact_corners": round(t5 - t4, 3),
            },
            "backend": str(jax.default_backend()),
        },
        "grid_lambda_sweep": {
            "n_samples": n_grid,
            "max_error": float(err_grid),
            "description": (
                f"Dense lambda grid: {args.n_lambda_grid} uniform points in "
                f"[-1, 1], {args.n_dm_grid} random dm per lambda."
            ),
        },
        "random_joint_sweep": {
            "n_samples": n_rand,
            "max_error": float(err_rand),
            "description": (
                f"Random joint: {args.n_lambda_rand} random lambda x "
                f"{args.n_dm_rand} random dm per lambda."
            ),
        },
        "boundary_stress": {
            "n_samples": n_bdy,
            "max_error": float(err_bdy),
            "description": (
                "lambda in {-1, -1+eps, -eps, 0, +eps, 1-eps, 1} with "
                f"{args.n_boundary} random dm each plus a deterministic "
                "dm-edge vector {-X, -X+eps, -eps, 0, +eps, X-eps, X}; "
                "eps = 1e-12."
            ),
        },
        "scale_sensitivity": {
            "n_samples": n_scale,
            "max_error": float(err_scale),
            "description": (
                f"{args.n_scale_samples} random (dm, lambda) pairs; S_r swept "
                f"log-uniformly over {args.n_scale_points} values in "
                "[max(|dm|, SCALE_TOL), max(|dm|, SCALE_TOL)*1e6]. "
                "Verifies that the decoded output continues to match "
                "lambda*dm at every admissible scale."
            ),
        },
        "exact_corners": {
            "n_samples": n_corners,
            "max_error": float(err_corners),
            "description": (
                "Deterministic 5x5 corner audit: (dm, lambda) on "
                "{-X, -X/2, 0, X/2, X} x {-1, -1/2, 0, 1/2, 1}."
            ),
        },
    }
    print()
    for k, v in results.items():
        if k == "setup":
            continue
        print(f"{k:22s}  n={v['n_samples']:>7d}  max_err={v['max_error']:.3e}")

    out = Path(__file__).resolve().parent / "synthetic_audit_table.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
