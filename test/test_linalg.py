import jax.numpy as jnp
from topo_dw.linalg import eigh, lowest_modes

def test_lowest_modes():
    H = jnp.diag(jnp.array([-1.0, 0.1, 0.2, 5.0]))
    Es, Vecs = lowest_modes(H, n_modes=2)
    assert Es.shape == (2,)
    assert Vecs.shape == (4, 2)
