
# API Reference

This document provides a structured overview of the public API  
for the **Topological Domain-Wall Fermions Framework (JAX)**.

For detailed explanations, see the main `API.md` in the repository root.

---

# Module Overview

The framework consists of the following modules:

- `mass_profiles` — domain-wall mass functions
- `hamiltonians` — lattice Hamiltonian builders
- `spectrum` — spectral analysis tools
- `interface` — interface-mode extraction
- `linalg` — linear algebra utilities
- `plotting` — visualization helpers

---

# 1. mass_profiles

### `mass_step_profile_y(Ny, m_top, m_bottom)`
Returns a step-like mass profile along y.

### `mass_kink_profile_y(Ny, m0=1.0, width=5.0)`
Returns a smooth tanh-kink mass profile.

---

# 2. hamiltonians

### `build_H_2d_wilson_kx(kx, Ny, m_y, r=1.0)`
Constructs the 2D Wilson–Dirac Hamiltonian in strip geometry.

Returns a Hermitian matrix of shape `(2*Ny, 2*Ny)`.

---

# 3. spectrum

### `spectrum_2d_wilson(m_y, Ny, Nkx=201, r=1.0)`
Computes the full band structure vs. momentum.

### `spectral_flow_2d_wilson(m_y, Ny, Nkx=101, n_modes=4, r=1.0)`
Extracts the lowest-|E| modes for each momentum.

---

# 4. interface

### `interface_modes_2d(m_y, Ny, kx_target=0.0, n_modes=4, r=1.0)`
Computes interface-localized modes and their spatial densities.

Returns:
- `ys` — y-coordinates  
- `Es` — energies  
- `Dens` — normalized densities  

---

# 5. linalg

### `eigh(H)`
Hermitian eigenvalue decomposition.

### `eigvals(H)`
Returns eigenvalues only.

### `eigvecs(H)`
Returns eigenvectors only.

### `lowest_modes(H, n_modes=4)`
Returns the n lowest-|E| modes.

### `batch_eigh(H_batch)`
Vectorized diagonalization for batches.

### `batch_lowest_modes(H_batch, n_modes=4)`
Vectorized lowest-mode extraction.

### `is_hermitian(H, tol=1e-10)`
Checks Hermiticity.

### `hermiticity_error(H)`
Returns maximum deviation from Hermiticity.

---

# 6. plotting

### `plot_spectrum_png(kxs, E, filename, title)`
Plots the full spectrum.

### `plot_interface_modes_png(ys, Es, Dens, filename, title)`
Plots interface-mode densities.

### `plot_spectral_flow_png(kxs, E_flow, filename, title)`
Plots spectral flow.

---

# Example Usage

```python
from topo_dw import *

Ny = 60
m = mass_kink_profile_y(Ny)

kxs, E = spectrum_2d_wilson(m, Ny)
ys, Es, Dens = interface_modes_2d(m, Ny)

plot_spectrum_png(kxs, E, "spectrum.png")
```
