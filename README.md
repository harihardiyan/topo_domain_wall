
Topological Domain‑Wall Fermions Framework (JAX)

A unified computational framework for 1D Jackiw–Rebbi, 2D Chern interfaces, and 2D Wilson–Dirac domain‑wall fermions

Author: Hari Hardiyan (AI Orchestration) ft. Microsoft CopilotGitHub: https://github.com/harihardiyan (github.com in Bing)

Overview

This repository provides a modular, JAX‑accelerated framework for studying topological fermions localized at domain walls in lattice systems.It unifies three major models used across condensed matter physics, high‑energy theory, and lattice gauge theory:

1D Jackiw–Rebbi domain‑wall fermion

2D Chern‑insulator interface modes

2D Wilson–Dirac fermions with domain‑wall mass

The framework is designed for:

numerical experiments

reproducible research

educational demonstrations

HPC scaling

rapid prototyping of new topological models

All computations are performed using pure JAX, ensuring differentiability, GPU/TPU compatibility, and high‑performance linear algebra.

Scientific Motivation

Domain‑wall fermions are a universal phenomenon appearing in:

topological insulators

quantum Hall interfaces

lattice QCD (Kaplan fermions)

anomaly inflow mechanisms

symmetry‑protected topological phases

A mass term that changes sign across space creates a topologically protected bound state.This framework provides a unified numerical environment to explore:

zero‑mode localization

spectral flow

robustness under disorder

Wilson mass regularization

chiral edge/interface modes

anomaly‑like behavior in lattice systems

Mathematical Background

1. Jackiw–Rebbi (1D)

The continuum Hamiltonian:

[ H = -i \sigma_x \partial_x + m(x)\sigma_z ]

A kink mass (m(x) = m_0 \tanh(x/w)) produces a zero‑energy bound state:

[ \psi_0(x) \propto \exp\left(-\int^x m(x') dx'\right) ]

The lattice version implemented here uses symmetric finite differences and open boundaries.

2. Chern‑insulator interface (2D)

A Qi–Wu–Zhang‑type Hamiltonian:

[ H(k_x, y) = \sin k_x \sigma_x + \sin k_y \sigma_y + (m(y) + \cos k_x + \cos k_y)\sigma_z ]

A domain wall in (m(y)) produces chiral interface modes with dispersion crossing the bulk gap.

3. Wilson–Dirac fermions (2D)

The lattice Dirac operator with Wilson term:

[ H = \sum_i \sin k_i \gamma_i + \left(m + r\sum_i (1 - \cos k_i)\right)\gamma_0 ]

In real‑space strip geometry (periodic in (x), open in (y)):

domain‑wall mass produces localized fermions

Wilson term removes fermion doubling

spectral flow reveals anomaly‑like behavior

This framework implements the full real‑space Hamiltonian:

[ H(k_x) \in \mathbb{C}^{(2N_y)\times(2N_y)} ]

and diagonalizes it for each (k_x).

Features

✔ 1D Jackiw–Rebbi zero mode

✔ 2D Chern interface modes

✔ 2D Wilson–Dirac domain‑wall fermions

✔ Spectral flow vs. momentum

✔ Interface‑mode spatial profiles

✔ Disorder robustness

✔ PNG visualization (spectrum, interface modes, spectral flow)

✔ Modular framework structure

✔ JAX‑accelerated linear algebra

✔ HPC‑ready design

Repository Structure

topo_domain_wall/
├── topo_dw/
│   ├── __init__.py
│   ├── mass_profiles.py
│   ├── hamiltonians.py
│   ├── spectrum.py
│   ├── interface.py
│   ├── plotting.py
├── run_demo_2d_wilson.py
├── README.md
└── API.md

Usage

Install dependencies

pip install jax jaxlib matplotlib

Run the 2D Wilson–Dirac demo

python run_demo_2d_wilson.py

This generates:

wilson2d_step_spectrum.png

wilson2d_step_interface_modes.png

wilson2d_step_spectral_flow.png

wilson2d_kink_spectrum.png

wilson2d_kink_interface_modes.png

wilson2d_kink_spectral_flow.png

Limitations

This framework is intentionally minimal and pedagogical.Current limitations include:

No gauge fields (U(1) or SU(3))

No interactions (pure single‑particle physics)

No finite‑temperature or Green’s function formalism

Matrix diagonalization scales as (O(N_y^3))

No automatic HPC parallelization (but JAX makes it easy to add)

These limitations are deliberate to keep the framework clean, transparent, and easy to extend.

API Documentation

See API.md for:

function signatures

module descriptions

mathematical definitions

usage examples

extension guidelines

Citation

If you use this framework in academic work, please cite:

Hari Hardiyan (AI Orchestration) ft. Microsoft Copilot,
"Topological Domain‑Wall Fermions Framework (JAX)",
https://github.com/harihardiyan/topo_domain_wall

Author

Hari Hardiyan AI Orchestration & Computational Physicsft. Microsoft Copilot

