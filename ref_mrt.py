"""Classical D3Q19 multiple-relaxation-time lattice Boltzmann collide--stream driver.

Linear moment relaxation on the same D3Q19 stencil and defaults as ``baseline_jax.py``.

Author: Muhammad Idrees Khan
Version: 1.0.0
"""

__version__ = "1.0.0"

import time

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

from viscous_dissipation import mean_dissipation_2nu_SijSij

# Default lattice and TGV demo parameters (``python ref_mrt.py`` entry point).
nx, ny, nz = 32, 32, 32
u0 = 0.03
tau = 0.505
nu = (tau - 0.5) / 3.0
dx = 2.0 * jnp.pi / nx
dt = dx * u0
iter_num = 2000

Q = 19
# D3Q19: 6 face + 12 diagonal + 1 rest link order (same as baseline_jax.py).
W = jnp.array([1 / 18] * 6 + [1 / 36] * 12 + [1 / 3])
Dirx = jnp.array([1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0])
Diry = jnp.array([0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 1, -1, 1, -1, 0, 0, 0, 0, 0])
Dirz = jnp.array([0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0])
NODE_VELOCITIES = jnp.array([Dirx, Diry, Dirz])

rho0 = 1.0

# D3Q19 moment matrix ``M`` and inverse; ``m = M f`` componentwise at each node.
def build_M_matrix_D3Q19():
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


M_matrix = build_M_matrix_D3Q19()
invM_matrix = jnp.linalg.inv(M_matrix)
s9_val = 1.0 / (3.0 * nu + 0.5)
s_vec = jnp.array(
    [
        0.0,
        1.19,
        1.4,
        0.0,
        1.2,
        0.0,
        1.2,
        0.0,
        1.2,
        s9_val,
        1.4,
        s9_val,
        1.4,
        s9_val,
        s9_val,
        s9_val,
        1.98,
        1.98,
        1.98,
    ]
)


@jax.jit
def get_equilibrium(u, rho):
    """Low-Mach D3Q19 equilibrium polynomial to O(u^2): ``rho * w * (1 + 3 eu + 4.5 eu^2 - 1.5 u^2)``, ``eu = e·u``."""
    eu = jnp.einsum("dQ,ijkd->ijkQ", NODE_VELOCITIES, u)
    u_sq = jnp.sum(u**2, axis=-1, keepdims=True)
    return rho[..., None] * W * (1 + 3 * eu + 4.5 * eu**2 - 1.5 * u_sq)


@jax.jit
def update(discrete_velocities_prev):
    """Advance one timestep: linear MRT collision then periodic streaming of distribution populations."""
    rho = jnp.sum(discrete_velocities_prev, axis=-1)
    rho_safe = jnp.where(rho < 1e-15, 1.0, rho)
    rhoinv = 1.0 / rho_safe
    u = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, discrete_velocities_prev) * rhoinv[
        ..., None
    ]
    jx = rho * u[..., 0]
    jy = rho * u[..., 1]
    jz = rho * u[..., 2]
    j_sq = jx**2 + jy**2 + jz**2
    delta_rho = rho - rho0

    m = jnp.einsum("ab,ijkb->ijka", M_matrix, discrete_velocities_prev)
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
    m_eq = m_eq.at[..., 16:19].set(0.0)

    m_post = m - jnp.einsum("ab,ijkb->ijka", jnp.diag(s_vec), (m - m_eq))
    f_post = jnp.einsum("ab,ijkb->ijka", invM_matrix, m_post)

    f_streamed = jnp.zeros_like(f_post)
    for i in range(Q):
        f_streamed = f_streamed.at[..., i].set(
            jnp.roll(
                jnp.roll(
                    jnp.roll(
                        f_post[..., i], NODE_VELOCITIES[0, i], axis=0
                    ),
                    NODE_VELOCITIES[1, i],
                    axis=1,
                ),
                NODE_VELOCITIES[2, i],
                axis=2,
            )
        )
    rho_streamed = jnp.sum(f_streamed, axis=-1)
    u_streamed = (
        jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, f_streamed)
        / rho_streamed[..., None]
    )
    return f_streamed, u_streamed


# Taylor--Green initial velocity (uses module-level nx, ny, nz, dx).
def tgv_velocity(u0):
    """TGV velocity field on module-level ``nx, ny, nz, dx``; same IC as ``run_ref_mrt``."""
    x = (jnp.arange(nx) + 0.5) * dx
    y = (jnp.arange(ny) + 0.5) * dx
    z = (jnp.arange(nz) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    ux = u0 * jnp.sin(X) * jnp.cos(Y) * jnp.cos(Z)
    uy = -u0 * jnp.cos(X) * jnp.sin(Y) * jnp.cos(Z)
    uz = jnp.zeros_like(X)
    return jnp.stack([ux, uy, uz], axis=-1)


rho_init = jnp.ones((nx, ny, nz))
u_init = tgv_velocity(u0)
discrete_velocities = get_equilibrium(u_init, rho_init)


def run_ref_mrt(
    nx_,
    ny_,
    nz_,
    u0_,
    tau_,
    timesteps_,
    quiet=True,
    collect_strain_dissipation=False,
):
    """Run Taylor--Green benchmark: return time, kinetic energy (NumPy arrays).

    If collect_strain_dissipation is True, also return the series from
    mean_dissipation_2nu_SijSij (viscous_dissipation module).
    """
    dx_ = 2.0 * jnp.pi / nx_
    dt_ = dx_ * u0_
    nu_ = (tau_ - 0.5) / 3.0
    s9_ = 1.0 / (3.0 * nu_ + 0.5)
    s_vec_ = jnp.array(
        [
            0.0,
            1.19,
            1.4,
            0.0,
            1.2,
            0.0,
            1.2,
            0.0,
            1.2,
            s9_,
            1.4,
            s9_,
            1.4,
            s9_,
            s9_,
            s9_,
            1.98,
            1.98,
            1.98,
        ]
    )
    M_ = build_M_matrix_D3Q19()
    invM_ = jnp.linalg.inv(M_)
    x_tgv = (jnp.arange(nx_) + 0.5) * dx_
    y_tgv = (jnp.arange(ny_) + 0.5) * dx_
    z_tgv = (jnp.arange(nz_) + 0.5) * dx_
    X_tgv, Y_tgv, Z_tgv = jnp.meshgrid(x_tgv, y_tgv, z_tgv, indexing="ij")
    ux_tgv = u0_ * jnp.sin(X_tgv) * jnp.cos(Y_tgv) * jnp.cos(Z_tgv)
    uy_tgv = -u0_ * jnp.cos(X_tgv) * jnp.sin(Y_tgv) * jnp.cos(Z_tgv)
    uz_tgv = jnp.zeros_like(X_tgv)
    u_init_ = jnp.stack([ux_tgv, uy_tgv, uz_tgv], axis=-1)
    rho_init_ = jnp.ones((nx_, ny_, nz_))
    discrete_velocities_ = get_equilibrium(u_init_, rho_init_)
    rho0_ = 1.0

    @jax.jit
    def update_local(dv_prev):
        rho = jnp.sum(dv_prev, axis=-1)
        rho_safe = jnp.where(rho < 1e-15, 1.0, rho)
        rhoinv = 1.0 / rho_safe
        u = jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, dv_prev) * rhoinv[..., None]
        jx = rho * u[..., 0]
        jy = rho * u[..., 1]
        jz = rho * u[..., 2]
        j_sq = jx**2 + jy**2 + jz**2
        delta_rho = rho - rho0_
        m = jnp.einsum("ab,ijkb->ijka", M_, dv_prev)
        m_eq = jnp.zeros_like(m)
        m_eq = m_eq.at[..., 0].set(delta_rho)
        m_eq = m_eq.at[..., 1].set(-11.0 * delta_rho + 19.0 * j_sq / rho0_)
        m_eq = m_eq.at[..., 2].set((-475.0 / 63.0) * j_sq / rho0_)
        m_eq = m_eq.at[..., 3].set(jx)
        m_eq = m_eq.at[..., 4].set(-(2.0 / 3.0) * jx)
        m_eq = m_eq.at[..., 5].set(jy)
        m_eq = m_eq.at[..., 6].set(-(2.0 / 3.0) * jy)
        m_eq = m_eq.at[..., 7].set(jz)
        m_eq = m_eq.at[..., 8].set(-(2.0 / 3.0) * jz)
        m_eq = m_eq.at[..., 9].set((2.0 * jx**2 - jy**2 - jz**2) / rho0_)
        m_eq = m_eq.at[..., 10].set(0.0)
        m_eq = m_eq.at[..., 11].set((jy**2 - jz**2) / rho0_)
        m_eq = m_eq.at[..., 12].set(0.0)
        m_eq = m_eq.at[..., 13].set(jx * jy / rho0_)
        m_eq = m_eq.at[..., 14].set(jy * jz / rho0_)
        m_eq = m_eq.at[..., 15].set(jx * jz / rho0_)
        m_eq = m_eq.at[..., 16:19].set(0.0)
        m_post = m - jnp.einsum("ab,ijkb->ijka", jnp.diag(s_vec_), (m - m_eq))
        f_post = jnp.einsum("ab,ijkb->ijka", invM_, m_post)
        f_streamed = jnp.zeros_like(f_post)
        for i in range(Q):
            f_streamed = f_streamed.at[..., i].set(
                jnp.roll(
                    jnp.roll(
                        jnp.roll(
                            f_post[..., i], NODE_VELOCITIES[0, i], axis=0
                        ),
                        NODE_VELOCITIES[1, i],
                        axis=1,
                    ),
                    NODE_VELOCITIES[2, i],
                    axis=2,
                )
            )
        rho_s = jnp.sum(f_streamed, axis=-1)
        u_s = (
            jnp.einsum("dQ,ijkQ->ijkd", NODE_VELOCITIES, f_streamed)
            / rho_s[..., None]
        )
        return f_streamed, u_s

    KE_list = []
    time_list = []
    eps_strain_list = []
    dv = discrete_velocities_
    it_range = range(timesteps_)
    if not quiet:
        it_range = tqdm(it_range, desc="ref_mrt")
    print_interval = max(1, timesteps_ // 100)
    for i in it_range:
        dv, u_s = update_local(dv)
        if collect_strain_dissipation:
            eps_strain_list.append(
                float(
                    jax.device_get(
                        mean_dissipation_2nu_SijSij(u_s, float(dx_), float(nu_))
                    )
                )
            )
        KE = float(
            jax.device_get(0.5 * jnp.mean(jnp.sum(u_s**2, axis=-1)))
        )
        KE_list.append(KE)
        time_list.append(float(i * dt_))
        if quiet and (i + 1) % print_interval == 0:
            print(f"  ref_mrt step {i + 1}/{timesteps_}")
    import numpy as _np

    out: list = [_np.array(time_list), _np.array(KE_list)]
    if collect_strain_dissipation:
        out.append(_np.asarray(eps_strain_list, dtype=_np.float64))
    if len(out) == 2:
        return out[0], out[1]
    return tuple(out)


if __name__ == "__main__":
    t0 = time.time()
    time_arr, ke_arr = run_ref_mrt(
        nx, ny, nz, u0, tau, iter_num, quiet=False
    )
    print(
        f"ref_mrt: {iter_num} steps in {time.time() - t0:.2f}s — "
        f"KE[0]={float(ke_arr[0]):.6e} KE[-1]={float(ke_arr[-1]):.6e}"
    )
