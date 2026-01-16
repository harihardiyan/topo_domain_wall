import jax.numpy as jnp
from .linalg import gamma_x, gamma_y, gamma_0

def build_H_2d_wilson_kx(kx, Ny, m_y, r=1.0):
    dim = 2 * Ny
    H = jnp.zeros((dim, dim), dtype=jnp.complex128)

    sin_kx = jnp.sin(kx)
    cos_kx = jnp.cos(kx)
    W_x = r * (1.0 - cos_kx)

    Hx_dirac = sin_kx * gamma_x
    Hx_wilson = W_x * gamma_0

    Hy_forward_dirac  = -0.5j * gamma_y
    Hy_backward_dirac =  0.5j * gamma_y
    Hy_forward_wilson  = -0.5 * r * gamma_0
    Hy_backward_wilson = -0.5 * r * gamma_0
    Hy_onsite_wilson   = r * gamma_0

    for j in range(Ny):
        i0 = 2 * j
        i1 = i0 + 2
        m_loc = m_y[j]
        onsite = m_loc * gamma_0 + Hx_dirac + Hx_wilson + Hy_onsite_wilson
        H = H.at[i0:i1, i0:i1].set(H[i0:i1, i0:i1] + onsite)

    for j in range(Ny - 1):
        i0 = 2 * j
        i1 = i0 + 2
        k0 = 2 * (j + 1)
        k1 = k0 + 2
        H = H.at[k0:k1, i0:i1].set(H[k0:k1, i0:i1] + Hy_forward_dirac + Hy_forward_wilson)
        H = H.at[i0:i1, k0:k1].set(H[i0:i1, k0:k1] + Hy_backward_dirac + Hy_backward_wilson)

    return H
