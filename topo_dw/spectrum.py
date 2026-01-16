import jax
import jax.numpy as jnp
from .hamiltonians import build_H_2d_wilson_kx

def spectrum_2d_wilson(m_y, Ny, Nkx=201, r=1.0):
    kxs = jnp.linspace(-jnp.pi, jnp.pi, Nkx)

    @jax.jit
    def evals_at_kx(kx):
        Hkx = build_H_2d_wilson_kx(kx, Ny, m_y, r=r)
        evals, _ = jnp.linalg.eigh(Hkx)
        return evals

    E = jax.vmap(evals_at_kx)(kxs)
    return kxs, E

def spectral_flow_2d_wilson(m_y, Ny, Nkx=101, n_modes=4, r=1.0):
    kxs = jnp.linspace(-jnp.pi, jnp.pi, Nkx)

    @jax.jit
    def low_evals_at_kx(kx):
        Hkx = build_H_2d_wilson_kx(kx, Ny, m_y, r=r)
        evals, _ = jnp.linalg.eigh(Hkx)
        idx_sorted = jnp.argsort(jnp.abs(evals))
        sel = idx_sorted[:n_modes]
        return evals[sel]

    E_low = jax.vmap(low_evals_at_kx)(kxs)
    return kxs, E_low
