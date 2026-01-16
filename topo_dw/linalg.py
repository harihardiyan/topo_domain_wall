import jax.numpy as jnp

# Konfigurasi presisi tinggi
import jax
jax.config.update("jax_enable_x64", True)

# Pauli Matrices
sx = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
sy = jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
sz = jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)

# Gamma shorthands
gamma_x = sx
gamma_y = sy
gamma_0 = sz
