"""
linalg.py
=========
High-level linear algebra utilities for the Topological Domain-Wall Fermions Framework.

This module provides:
- Hermitian eigenvalue solvers
- Lowest-mode extraction
- Batched diagonalization (vmap)
- JIT-compiled linear algebra routines
- Diagnostics (Hermiticity check, norm check)

All functions are JAX-friendly and designed for HPC scaling.
"""

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


# ============================================================
# Hermiticity diagnostics
# ============================================================

def is_hermitian(H, tol=1e-10):
    """
    Check whether a matrix H is Hermitian: H = H^\dagger.
    Returns True/False.
    """
    diff = jnp.max(jnp.abs(H - jnp.conj(H.T)))
    return diff < tol


def hermiticity_error(H):
    """
    Return the maximum deviation from Hermiticity.
    Useful for debugging Hamiltonian construction.
    """
    return jnp.max(jnp.abs(H - jnp.conj(H.T)))


# ============================================================
# Eigenvalue solvers
# ============================================================

@jax.jit
def eigh(H):
    """
    Hermitian eigenvalue decomposition.
    Wrapper around jnp.linalg.eigh with JIT.
    Returns (evals, evecs).
    """
    return jnp.linalg.eigh(H)


@jax.jit
def eigvals(H):
    """
    Return only eigenvalues of a Hermitian matrix.
    """
    evals, _ = jnp.linalg.eigh(H)
    return evals


@jax.jit
def eigvecs(H):
    """
    Return only eigenvectors of a Hermitian matrix.
    """
    _, evecs = jnp.linalg.eigh(H)
    return evecs


# ============================================================
# Lowest-mode extraction
# ============================================================

def lowest_modes(H, n_modes=4):
    """
    Extract the n lowest-energy modes (by |E|) of a Hermitian matrix H.

    Returns:
        Es:   shape (n_modes,)
        Vecs: shape (dim, n_modes)
    """
    evals, evecs = jnp.linalg.eigh(H)
    idx = jnp.argsort(jnp.abs(evals))[:n_modes]
    return evals[idx], evecs[:, idx]


# ============================================================
# Batched diagonalization (for k-space scans)
# ============================================================

def batch_eigh(H_batch):
    """
    Diagonalize a batch of Hermitian matrices.
    H_batch: shape (B, N, N)

    Returns:
        evals: shape (B, N)
        evecs: shape (B, N, N)
    """
    return jax.vmap(jnp.linalg.eigh)(H_batch)


def batch_lowest_modes(H_batch, n_modes=4):
    """
    Extract lowest |E| modes for a batch of Hamiltonians.

    Returns:
        Es:   shape (B, n_modes)
        Vecs: shape (B, N, n_modes)
    """
    def _one(H):
        evals, evecs = jnp.linalg.eigh(H)
        idx = jnp.argsort(jnp.abs(evals))[:n_modes]
        return evals[idx], evecs[:, idx]

    Es, Vecs = jax.vmap(_one)(H_batch)
    return Es, Vecs


# ============================================================
# Norm utilities
# ============================================================

def normalize(vec):
    """
    Normalize a complex vector.
    """
    return vec / jnp.sqrt(jnp.vdot(vec, vec))


def normalize_batch(vecs):
    """
    Normalize a batch of vectors.
    vecs: shape (B, N)
    """
    norms = jnp.sqrt(jnp.sum(jnp.abs(vecs)**2, axis=1))
    return vecs / norms[:, None]
