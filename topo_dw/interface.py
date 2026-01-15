import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .hamiltonians import build_H_2d_wilson_kx

def interface_modes_2d(m_y, Ny, kx_target=0.0, n_modes=4, r=1.0):
    Hkx = build_H_2d_wilson_kx(kx_target, Ny, m_y, r=r)
    evals, evecs = jnp.linalg.eigh(Hkx)
    idx_sorted = jnp.argsort(jnp.abs(evals))

    Es = []
    Dens_list = []

    for n in range(int(n_modes)):
        idx = int(idx_sorted[n])
        e = evals[idx]
        v = evecs[:, idx]

        dens = []
        for j in range(Ny):
            i0 = 2 * j
            i1 = i0 + 2
            psi_j = v[i0:i1]
            dens_j = (jnp.vdot(psi_j, psi_j)).real
            dens.append(dens_j)
        dens = jnp.array(dens)
        dens = dens / jnp.sum(dens)

        Es.append(e)
        Dens_list.append(dens)

    Es = jnp.array(Es)
    Dens = jnp.stack(Dens_list, axis=0)
    ys = jnp.arange(Ny)
    return ys, Es, Dens
