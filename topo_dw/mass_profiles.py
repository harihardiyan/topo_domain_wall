import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def mass_step_profile_y(Ny, m_top, m_bottom):
    half = Ny // 2
    m = jnp.ones(Ny) * m_bottom
    m = m.at[:half].set(m_top)
    return m


def mass_kink_profile_y(Ny, m0=1.0, width=5.0):
    y = jnp.linspace(-Ny / 2, Ny / 2, Ny)
    return m0 * jnp.tanh(y / width)
