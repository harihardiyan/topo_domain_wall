"""
TDWF-JAX: Topological Domain-Wall Fermions Framework
----------------------------------------------------
A hardware-accelerated, unified framework for 1D Jackiw-Rebbi,
2D Chern interfaces, and 2D Wilson-Dirac domain-wall fermions.
"""

__version__ = "1.0.0"

from .spectrum import spectrum_2d_wilson, spectral_flow_2d_wilson
from .interface import interface_modes_2d
from .hamiltonians import build_H_2d_wilson_kx
from .mass_profiles import mass_step_profile_y, mass_kink_profile_y
from .plotting import plot_spectrum, plot_interface_modes, plot_spectral_flow

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
