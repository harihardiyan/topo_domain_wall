import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .hamiltonians import build_H_2d_wilson_kx


def interface_modes_2d(m_y, Ny, kx_target=0.0, n_modes=4, r=1.0):
    H = build_H_2d_wilson_kx(kx_target, Ny, m_y, r=r)
    evals, evecs = jnp.linalg.eigh(H)
    idx = jnp.argsort(jnp.abs(evals))[:n_modes]
    Es = evals[idx]
    Vecs = evecs[:, idx]

    Dens = jnp.abs(Vecs.reshape((Ny, 2, n_modes))) ** 2
    Dens_sum = jnp.sum(Dens, axis=1)
    Dens_norm = Dens_sum / jnp.sum(Dens_sum, axis=1, keepdims=True)

    ys = jnp.arange(Ny)
    return ys, Es, Dens_norm
