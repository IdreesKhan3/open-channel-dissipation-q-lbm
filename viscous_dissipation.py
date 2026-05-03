"""Volume mean of viscous dissipation 2 nu <S_ij S_ij> from a velocity field u.

Symmetric strain tensor S_ij = ((du_j/dx_i) + (du_i/dx_j)) / 2 with second-order
central differences on a periodic stencil (jax.numpy.roll). Shared by ``ref_mrt.py``
and ``baseline_jax.py`` for dissipation diagnostics.

Author: Muhammad Idrees Khan
Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

import jax.numpy as jnp


def mean_dissipation_2nu_SijSij(u, dx, nu):
    """Return 2 * nu * spatial_mean( sum_ij S_ij S_ij ). u shape (nx, ny, nz, 3)."""
    dx = jnp.asarray(dx, dtype=u.dtype)
    nu = jnp.asarray(nu, dtype=u.dtype)
    dudxi = []
    for a in range(3):
        row = []
        for c in range(3):
            du = (
                jnp.roll(u[..., c], -1, axis=a) - jnp.roll(u[..., c], 1, axis=a)
            ) / (2.0 * dx)
            row.append(du)
        dudxi.append(jnp.stack(row, axis=-1))
    grad = jnp.stack(dudxi, axis=-1)
    S = 0.5 * (grad + jnp.swapaxes(grad, -1, -2))
    s2 = jnp.sum(S * S, axis=(-2, -1))
    return 2.0 * nu * jnp.mean(s2)
