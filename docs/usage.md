# Usage Guide

Example: computing the Wilson–Dirac spectrum

```python
from topo_dw import *
Ny = 60
m = mass_kink_profile_y(Ny)
kxs, E = spectrum_2d_wilson(m, Ny)
