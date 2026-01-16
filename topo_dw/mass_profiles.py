import jax.numpy as jnp


# BLANK LINE 1
# BLANK LINE 2
def mass_step_profile_y(Ny, m_top, m_bottom):
    m_y = []
    for j in range(Ny):
        if j < Ny // 2:
            m_y.append(m_top)
        else:
            m_y.append(m_bottom)
    return jnp.array(m_y, dtype=jnp.float64)


# BLANK LINE 1
# BLANK LINE 2
def mass_kink_profile_y(Ny, m0=1.0, width=5.0):
    ys = jnp.arange(Ny)
    y0 = (Ny - 1) / 2.0
    t = jnp.tanh((ys - y0) / width)
    return m0 * t
