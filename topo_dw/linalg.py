import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def is_hermitian(H, tol=1e-10):
    diff = jnp.max(jnp.abs(H - jnp.conj(H.T)))
    return diff < tol


def hermiticity_error(H):
    return jnp.max(jnp.abs(H - jnp.conj(H.T)))


def eigh(H):
    return jnp.linalg.eigh(H)


def eigvals(H):
    evals, _ = jnp.linalg.eigh(H)
    return evals


def eigvecs(H):
    _, evecs = jnp.linalg.eigh(H)
    return evecs


def lowest_modes(H, n_modes=4):
    evals, evecs = jnp.linalg.eigh(H)
    idx = jnp.argsort(jnp.abs(evals))[:n_modes]
    return evals[idx], evecs[:, idx]


def batch_eigh(H_batch):
    return jax.vmap(jnp.linalg.eigh)(H_batch)


def batch_lowest_modes(H_batch, n_modes=4):
    def _one(H):
        evals, evecs = jnp.linalg.eigh(H)
        idx = jnp.argsort(jnp.abs(evals))[:n_modes]
        return evals[idx], evecs[:, idx]

    return jax.vmap(_one)(H_batch)


def normalize(vec):
    return vec / jnp.sqrt(jnp.vdot(vec, vec))


def normalize_batch(vecs):
    norms = jnp.sqrt(jnp.sum(jnp.abs(vecs) ** 2, axis=1))
    return vecs / norms[:, None]
