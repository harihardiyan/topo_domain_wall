from topo_dw.hamiltonians import build_H_2d_wilson_kx
from topo_dw.linalg import is_hermitian
import jax.numpy as jnp

def test_hermitian():
    Ny = 20
    m = jnp.zeros(Ny)
    H = build_H_2d_wilson_kx(0.1, Ny, m)
    assert is_hermitian(H)
