import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def build_H_2d_wilson_kx(kx, Ny, m_y, r=1.0):
    dim = Ny
    H = jnp.zeros((2 * dim, 2 * dim), dtype=jnp.complex128)

    for y in range(dim):
        m = m_y[y]
        idx = 2 * y
        H = H.at[idx, idx].set(m + r * 2)
        H = H.at[idx + 1, idx + 1].set(-m - r * 2)

        if y < dim - 1:
            H = H.at[idx, idx + 2].set(-r)
            H = H.at[idx + 1, idx + 3].set(-r)
            H = H.at[idx + 2, idx].set(-r)
            H = H.at[idx + 3, idx + 1].set(-r)

            H = H.at[idx, idx + 3].set(-1j / 2)
            H = H.at[idx + 1, idx + 2].set(-1j / 2)
            H = H.at[idx + 2, idx + 1].set(1j / 2)
            H = H.at[idx + 3, idx].set(1j / 2)

    for y in range(dim):
        idx = 2 * y
        H = H.at[idx, idx + 1].set(jnp.sin(kx))
        H = H.at[idx + 1, idx].set(jnp.sin(kx))

    return H
