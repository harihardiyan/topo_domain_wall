
Topological Domain‑Wall Fermions Framework (JAX)

Author: Hari Hardiyan (AI Orchestration) ft. Microsoft CopilotRepository: https://github.com/harihardiyan/topo_domain_wall (github.com in Bing)

📘 API Documentation

This document describes the public API of the topo_dw framework.It covers:

Module structure

Function descriptions

Input/output specifications

Mathematical notes

Usage examples

Best practices

All functions are implemented using JAX, ensuring:

Pure functional behavior

JIT compilation

GPU/TPU compatibility

Vectorized operations via vmap

------------------------------------------------------------

🧩 Module: mass_profiles

------------------------------------------------------------

### mass_step_profile_y(Ny, m_top, m_bottom)

Generates a step‑like domain‑wall mass profile along the y‑direction.

Parameters

Ny (int) — number of lattice sites along y

m_top (float) — mass value for the upper region

m_bottom (float) — mass value for the lower region

Returnsjnp.ndarray of shape (Ny,)

Example

m_y = mass_step_profile_y(60, -1.0, 1.0)

### mass_kink_profile_y(Ny, m0=1.0, width=5.0)

Generates a smooth tanh‑kink mass profile:

[ m(y) = m_0 \tanh\left(\frac{y - y_0}{\text{width}}\right) ]

Returnsjnp.ndarray of shape (Ny,)

------------------------------------------------------------

🧩 Module: hamiltonians

------------------------------------------------------------

### build_H_2d_wilson_kx(kx, Ny, m_y, r=1.0)

Constructs the 2D Wilson–Dirac Hamiltonian in strip geometry:

Periodic in x (via momentum kx)

Open boundary in y

Includes Wilson mass term to remove fermion doubling

Parameters

kx (float) — momentum along x

Ny (int) — number of y‑sites

m_y (array) — mass profile along y

r (float) — Wilson parameter

ReturnsHermitian matrix H of shape (2*Ny, 2*Ny)

------------------------------------------------------------

🧩 Module: spectrum

------------------------------------------------------------

### spectrum_2d_wilson(m_y, Ny, Nkx=201, r=1.0)

Computes the full energy spectrum of the Wilson–Dirac Hamiltonian as a function of kx.

Returns

kxs — array (Nkx,)

E — array (Nkx, 2*Ny)

Example

kxs, E = spectrum_2d_wilson(m_y, Ny=60)

### spectral_flow_2d_wilson(m_y, Ny, Nkx=101, n_modes=4, r=1.0)

Extracts the lowest |E| modes for each kx, used to visualize spectral flow.

Returns

kxs — array (Nkx,)

E_low — array (Nkx, n_modes)

------------------------------------------------------------

🧩 Module: interface

------------------------------------------------------------

### interface_modes_2d(m_y, Ny, kx_target=0.0, n_modes=4, r=1.0)

Extracts interface‑localized modes at a specific momentum.

Returns

ys — array of y‑coordinates

Es — energies of selected modes (n_modes,)

Dens — spatial density profiles (n_modes, Ny)

Example

ys, Es, Dens = interface_modes_2d(m_y, Ny=60, kx_target=0.0)

------------------------------------------------------------

🧩 Module: linalg

------------------------------------------------------------

### eigh(H)

JIT‑compiled Hermitian eigenvalue decomposition.

Returnsevals, evecs

### eigvals(H)

Returns only eigenvalues.

### eigvecs(H)

Returns only eigenvectors.

### lowest_modes(H, n_modes=4)

Extracts the n modes with smallest |E|.

Returns

Es — (n_modes,)

Vecs — (dim, n_modes)

### batch_eigh(H_batch)

Diagonalizes a batch of Hamiltonians.

Returns

evals — (B, N)

evecs — (B, N, N)

### batch_lowest_modes(H_batch, n_modes=4)

Extracts lowest modes for each Hamiltonian in a batch.

### is_hermitian(H, tol=1e-10)

Checks Hermiticity.

### hermiticity_error(H)

Returns maximum deviation from Hermiticity.

------------------------------------------------------------

🧩 Module: plotting

------------------------------------------------------------

### plot_spectrum_png(kxs, E, filename, title)

Plots the full spectrum vs. momentum.

### plot_interface_modes_png(ys, Es, Dens, filename, title)

Plots interface‑mode density profiles.

### plot_spectral_flow_png(kxs, E_flow, filename, title)

Plots spectral flow of low‑energy modes.

------------------------------------------------------------

🧪 Example Workflow

------------------------------------------------------------

from topo_dw import *

Ny = 60
m_y = mass_kink_profile_y(Ny)

# Compute spectrum
kxs, E = spectrum_2d_wilson(m_y, Ny)

# Extract interface modes
ys, Es, Dens = interface_modes_2d(m_y, Ny, kx_target=0.0)

# Plot results
plot_spectrum_png(kxs, E, "spectrum.png")

------------------------------------------------------------

📌 Best Practices

------------------------------------------------------------

Always check Hermiticity when modifying Hamiltonians

Use batch_eigh for k‑space scans

Use jax.jit for repeated computations

Use vmap for parallel diagonalization

For HPC, combine jit + pmap

------------------------------------------------------------

📖 Citation

------------------------------------------------------------

@misc{hardiyan_topodwf_2026,
  author       = {Hari Hardiyan and Microsoft Copilot},
  title        = {Topological Domain-Wall Fermions Framework (JAX)},
  year         = {2026},
  howpublished = {\url{https://github.com/harihardiyan/topo_domain_wall}},
  note         = {AI Orchestration & Computational Physics}
}

