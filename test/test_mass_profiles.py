import jax.numpy as jnp
from topo_dw.mass_profiles import mass_step_profile_y, mass_kink_profile_y

def test_step_profile():
    m = mass_step_profile_y(10, -1, 1)
    assert m.shape == (10,)
    assert jnp.all(m[:5] == -1)
    assert jnp.all(m[5:] == 1)

def test_kink_profile():
    m = mass_kink_profile_y(10)
    assert m.shape == (10,)
    assert jnp.all(jnp.isfinite(m))
