"""
TDWF-JAX: Topological Domain-Wall Fermions Framework
----------------------------------------------------
A hardware-accelerated, unified framework for the investigation of 
1D Jackiw–Rebbi, 2D Chern interfaces, and 2D Wilson–Dirac 
domain-wall fermions using JAX and XLA-optimized linear algebra.

Author: Hari Hardiyan
Affiliation: AI Orchestration & Computational Physics
License: MIT
"""

__version__ = "1.0.0"

# --- Core Spectral Solvers & Analysis ---
from .spectrum import (
    spectrum_2d_wilson,
    spectral_flow_2d_wilson
)

# --- Interface & Localization Analysis ---
from .interface import (
    interface_modes_2d
)

# --- Hamiltonian & Lattice Construction ---
from .hamiltonians import (
    build_H_2d_wilson_kx
)

# --- Spatial Mass Profiling Suite ---
from .mass_profiles import (
    mass_step_profile_y,
    mass_kink_profile_y
)

# --- Scientific Visualization Suite ---
from .plotting import (
    plot_spectrum,
    plot_interface_modes,
    plot_spectral_flow
)

# Defining public API exposure
__all__ = [
    "spectrum_2d_wilson",
    "spectral_flow_2d_wilson",
    "interface_modes_2d",
    "build_H_2d_wilson_kx",
    "mass_step_profile_y",
    "mass_kink_profile_y",
    "plot_spectrum",
    "plot_interface_modes",
    "plot_spectral_flow"
]
