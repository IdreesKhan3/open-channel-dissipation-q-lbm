"""D3Q19 MRT collide--stream with CPTP dissipative relaxation (Kraus) and adaptive S_r.

Taylor--Green setup; classical linear counterpart in ``ref_mrt.py``.

Author: Muhammad Idrees Khan
Version: 1.0.0
"""

from __future__ import annotations

__version__ = "1.0.0"

import numpy as np

import jax
import jax.numpy as jnp

from viscous_dissipation import mean_dissipation_2nu_SijSij

jax.config.update("jax_enable_x64", True)

Q = 19
# D3Q19: 6 face + 12 diagonal + 1 rest link order (same as ref_mrt.py).
W = jnp.array([1 / 18] * 6 + [1 / 36] * 12 + [1 / 3])
Dirx = jnp.array([1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0])
Diry = jnp.array([0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0])
Dirz = jnp.array([0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0])
NODE_VELOCITIES = jnp.array([Dirx, Diry, Dirz])

rho0 = 1.0
# Rows 0,3,5,7: no linear relaxation; remaining non-conserved rows use CPTP in DISSIPATIVE.
CONSERVED_MOMENT_MASK = jnp.array(
    [True] + [False] * 2 + [True] + [False] + [True] + [False] + [True] + [False] * 11
)
DISSIPATIVE = (1, 2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18)
SCALE_TOL = 1e-12


def build_M_matrix_D3Q19():
    """Moment matrix mapping population vector f to moment vector m (D3Q19)."""
    e_alpha = jnp.stack([Dirx, Diry, Dirz], axis=1)
    M = jnp.zeros((19, 19))
    for alpha in range(19):
        ex, ey, ez = e_alpha[alpha]
        e_norm_sq = ex * ex + ey * ey + ez * ez
        M = M.at[0, alpha].set(1.0)
        M = M.at[1, alpha].set(19 * e_norm_sq - 30.0)
        M = M.at[2, alpha].set(0.5 * (21 * e_norm_sq**2 - 53 * e_norm_sq + 24.0))
        M = M.at[3, alpha].set(ex)
        M = M.at[4, alpha].set((5 * e_norm_sq - 9.0) * ex)
        M = M.at[5, alpha].set(ey)
        M = M.at[6, alpha].set((5 * e_norm_sq - 9.0) * ey)
        M = M.at[7, alpha].set(ez)
        M = M.at[8, alpha].set((5 * e_norm_sq - 9.0) * ez)
        M = M.at[9, alpha].set(3 * ex * ex - e_norm_sq)
        M = M.at[10, alpha].set((3 * e_norm_sq - 5.0) * (3 * ex * ex - e_norm_sq))
        M = M.at[11, alpha].set(ey * ey - ez * ez)
        M = M.at[12, alpha].set((3 * e_norm_sq - 5.0) * (ey * ey - ez * ez))
        M = M.at[13, alpha].set(ex * ey)
        M = M.at[14, alpha].set(ey * ez)
        M = M.at[15, alpha].set(ex * ez)
        M = M.at[16, alpha].set((ey * ey - ez * ez) * ex)
        M = M.at[17, alpha].set((ez * ez - ex * ex) * ey)
        M = M.at[18, alpha].set((ex * ex - ey * ey) * ez)
    return M


def get_equilibrium(u, rho):
    """Low-Mach D3Q19 equilibrium polynomial to O(u^2): ``rho * w * (1 + 3 eu + 4.5 eu^2 - 1.5 u^2)``, ``eu = e·u``."""
    eu = jnp.einsum("dQ,ijkd->ijkQ", NODE_VELOCITIES, u)
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    return rho[..., None] * W * (1 + 3 * eu + 4.5 * eu**2 - 1.5 * u_sq)


def _apply_kraus_jax(dm_r, S_r, kraus):
    """Signed two-rail populations, apply four CPTP Kraus operators (each ``4×4``), decode to scalar."""
    p_plus = jnp.maximum(dm_r, 0.0) / S_r
    p_minus = jnp.maximum(-dm_r, 0.0) / S_r
    rho_plus = jnp.zeros((*dm_r.shape, 2, 2), dtype=jnp.complex128)
    rho_plus = rho_plus.at[..., 0, 0].set(1.0 - p_plus).at[..., 1, 1].set(p_plus)
    rho_minus = jnp.zeros((*dm_r.shape, 2, 2), dtype=jnp.complex128)
    rho_minus = rho_minus.at[..., 0, 0].set(1.0 - p_minus).at[..., 1, 1].set(p_minus)
    rho = jnp.einsum("...ij,...kl->...ikjl", rho_plus, rho_minus).reshape((*dm_r.shape, 4, 4))
    rho_out = jnp.zeros_like(rho)
    for k in range(4):
        K = kraus[k]
        rho_out = rho_out + jnp.einsum("ij,...jk,kl->...il", K, rho, jnp.conj(K).T)
    n_plus = jnp.kron(jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128), jnp.eye(2, dtype=jnp.complex128))
    n_minus = jnp.kron(jnp.eye(2, dtype=jnp.complex128), jnp.array([[0, 0], [0, 1]], dtype=jnp.complex128))
    p_plus_out = jnp.real(jnp.einsum("ij,...ji->...", n_plus, rho_out))
    p_minus_out = jnp.real(jnp.einsum("ij,...ji->...", n_minus, rho_out))
    return S_r * (p_plus_out - p_minus_out)


def _build_kraus(alpha, need_swap):
    """Tensor product of single-rail AD Kraus; optional rail SWAP if need_swap."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(alpha)]], dtype=np.complex128)
    K1 = np.array([[0.0, np.sqrt(1.0 - alpha)], [0.0, 0.0]], dtype=np.complex128)
    kraus = [
        np.kron(K0, K0),
        np.kron(K0, K1),
        np.kron(K1, K0),
        np.kron(K1, K1),
    ]
    if need_swap:
        SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.complex128)
        kraus = [SWAP @ K for K in kraus]
    return jnp.array(kraus)

def run_baseline_open_channel_jax(
    nx,
    ny,
    nz,
    u0,
    tau,
    timesteps,
    quiet=True,
    debug_history=None,
    collect_strain_dissipation=False,
):
    """Run Taylor--Green benchmark; return arrays (time, kinetic energy).

    If collect_strain_dissipation is True, append the viscous dissipation series
    (mean_dissipation_2nu_SijSij). If debug_history is a non-None list, append
    per-step max-errors comparing CPTP moments to linear MRT targets.
    """
    dx = 2.0 * jnp.pi / nx
    dt = 2.0 * np.pi / nx * u0
    nu = (tau - 0.5) / 3.0
    s9 = 1.0 / (3.0 * nu + 0.5)
    s_vec_np = np.array([
        0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
        s9, 1.4, s9, 1.4, s9, s9, s9, 1.98, 1.98, 1.98
    ], dtype=np.float64)
    s_vec = jnp.array(s_vec_np)
    lam = 1.0 - s_vec
    # Linear MRT multipliers lambda_r (used only for dm_target / err_* diagnostics).
    scale = jnp.where(CONSERVED_MOMENT_MASK, 1.0, lam)
    M_ = build_M_matrix_D3Q19()
    invM_ = jnp.linalg.inv(M_)

    # TGV initial velocity; lattice step matches ``ref_mrt.run_ref_mrt`` (``dx = 2π/nx``, ``dt = dx * u0``).
    x_tgv = (jnp.arange(nx) + 0.5) * dx
    y_tgv = (jnp.arange(ny) + 0.5) * dx
    z_tgv = (jnp.arange(nz) + 0.5) * dx
    X_tgv, Y_tgv, Z_tgv = jnp.meshgrid(x_tgv, y_tgv, z_tgv, indexing="ij")
    ux_tgv = u0 * jnp.sin(X_tgv) * jnp.cos(Y_tgv) * jnp.cos(Z_tgv)
    uy_tgv = -u0 * jnp.cos(X_tgv) * jnp.sin(Y_tgv) * jnp.cos(Z_tgv)
    uz_tgv = jnp.zeros_like(X_tgv)
    u_init = jnp.stack([ux_tgv, uy_tgv, uz_tgv], axis=-1)
    rho_init = jnp.ones((nx, ny, nz))
    dv_init = get_equilibrium(u_init, rho_init)

    # One Kraus tuple per dissipative moment; lambda_r < 0 => rail SWAP in _build_kraus.
    kraus_list = []
    for r in DISSIPATIVE:
        s_r = float(s_vec_np[r])
        lam_r = 1.0 - s_r
        alpha_r = abs(lam_r)
        need_swap_r = lam_r < 0.0
        kraus_list.append(_build_kraus(alpha_r, need_swap_r))
    kraus_stack = jnp.stack(kraus_list)

    @jax.jit
    def update_local(dv_prev):
        # --- macro -> moments m, equilibrium moments m_eq ---
        rho = jnp.sum(dv_prev, axis=-1)
        rho_safe = jnp.where(rho < 1e-15, 1.0, rho)
        rhoinv = 1.0 / rho_safe
        u = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, dv_prev) * rhoinv[..., None]
        jx = rho * u[..., 0]
        jy = rho * u[..., 1]
        jz = rho * u[..., 2]
        j_sq = jx**2 + jy**2 + jz**2
        delta_rho = rho - rho0
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
        dm = m - m_eq
        # Collision: Kraus CPTP on each dissipative ``dm_r`` (decoded equivalent to ``λ_r dm_r``).
        dm_prime = dm
        for idx, r in enumerate(DISSIPATIVE):
            kraus_r = kraus_stack[idx]
            dm_r = dm[..., r]
            S_r = jnp.maximum(jnp.abs(dm_r), SCALE_TOL)
            cptp_val = _apply_kraus_jax(dm_r, S_r, kraus_r)
            dm_prime = dm_prime.at[..., r].set(cptp_val)
        m_plus = m_eq + dm_prime
        f_post = jnp.einsum("ab,ijkb->ijka", invM_, m_plus)

        # Reference: classical linear relaxation dm*scale; compare to dm_prime above.
        dm_target = dm * scale[None, None, None, :]
        m_plus_target = m_eq + dm_target
        f_post_target = jnp.einsum("ab,ijkb->ijka", invM_, m_plus_target)
        err_dm_max = jnp.max(jnp.abs(dm_prime - dm_target))
        err_mplus_max = jnp.max(jnp.abs(m_plus - m_plus_target))
        err_fpost_max = jnp.max(jnp.abs(f_post - f_post_target))

        # --- stream: post-collision populations ---
        f_streamed = jnp.zeros_like(f_post)
        for i in range(Q):
            f_streamed = f_streamed.at[..., i].set(
                jnp.roll(jnp.roll(jnp.roll(f_post[..., i], NODE_VELOCITIES[0, i], axis=0),
                                 NODE_VELOCITIES[1, i], axis=1),
                         NODE_VELOCITIES[2, i], axis=2)
            )
        rho_s = jnp.sum(f_streamed, axis=-1)
        mom_s = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, f_streamed)
        u_stream = mom_s / rho_s[..., None]
        return f_streamed, u_stream, err_dm_max, err_mplus_max, err_fpost_max

    dx_f = float(2.0 * np.pi / nx)
    nu_f = float((tau - 0.5) / 3.0)

    KE_list = []
    time_list = []
    eps_strain_list = []
    dv = dv_init
    print_interval = max(1, timesteps // 100)
    record_debug = debug_history is not None
    for i in range(timesteps):
        dv, u_stream, err_dm, err_mplus, err_fpost = update_local(dv)
        if record_debug:
            debug_history.append({
                "step": i + 1,
                "err_dm_max": float(jax.device_get(err_dm)),
                "err_mplus_max": float(jax.device_get(err_mplus)),
                "err_fpost_max": float(jax.device_get(err_fpost)),
            })
        if collect_strain_dissipation:
            eps_strain_list.append(
                float(
                    jax.device_get(
                        mean_dissipation_2nu_SijSij(u_stream, dx_f, nu_f)
                    )
                )
            )
        KE = float(jax.device_get(0.5 * jnp.mean(jnp.sum(u_stream**2, axis=-1))))
        KE_list.append(KE)
        time_list.append(float(i * dt))
        if quiet and (i + 1) % print_interval == 0:
            print(f"  BaseLine open-channel step {i + 1}/{timesteps}")
    out = [np.array(time_list), np.array(KE_list)]
    if collect_strain_dissipation:
        out.append(np.asarray(eps_strain_list, dtype=np.float64))
    if len(out) == 2:
        return out[0], out[1]
    return tuple(out)
