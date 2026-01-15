
# TDWF-JAX: Topological Domain-Wall Fermions Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)](https://www.python.org/downloads/)
[![Framework: JAX](https://img.shields.io/badge/Framework-JAX-orange.svg)](https://github.com/google/jax)
[![Field: Computational Physics](https://img.shields.io/badge/Field-Condensed--Matter-blue)](#)

## 📦 Overview

This repository provides a modular, **JAX-accelerated computational framework** dedicated to the study of topological fermions localized at domain walls within lattice systems. It unifies three seminal models across condensed matter physics, high-energy theory, and lattice gauge theory:

1.  **1D Jackiw–Rebbi:** Solitons and zero-mode fermions.
2.  **2D Chern Insulators:** Interface modes at topological phase boundaries.
3.  **2D Wilson–Dirac:** Domain-wall mass regularization on lattice geometries.

**Designed for:**
*   🧪 **Numerical Experiments:** High-precision simulation of topological phases.
*   📚 **Reproducible Research:** Audit-grade code for scientific publications.
*   🎓 **Pedagogical Demonstrations:** Transparent implementation of complex topological concepts.
*   🚀 **HPC Scaling:** Ready for GPU/TPU environments via XLA compilation.
*   🧠 **Rapid Prototyping:** Flexible API for developing novel topological models.

All computations utilize **pure JAX**, ensuring full differentiability, hardware acceleration, and optimized linear algebra performance.

---

## 🧠 Scientific Motivation

Domain-wall fermions represent a universal physical phenomenon, appearing critically in:
*   **Topological Insulators:** Symmetry-protected conduction at material interfaces.
*   **Quantum Hall Systems:** Robustness of chiral edge currents.
*   **Lattice QCD:** Implementation of **Kaplan fermions** to preserve chiral symmetry.
*   **Anomaly Inflow:** Compensation of bulk anomalies through boundary-localized modes.
*   **Topological Phase Transitions:** Dynamics of fermions across spatially varying mass profiles.

---

## 🧮 Mathematical Formulation

### 🔹 Jackiw–Rebbi (1D)
$$H = -i \sigma_x \partial_x + m(x)\sigma_z$$
A kink mass profile, $m(x) = m_0 \tanh(x/w)$, produces a topologically protected zero-energy bound state defined by:
$$\psi_0(x) \propto \exp\left(-\int^x m(x') dx'\right)$$

### 🔹 Chern Interface (2D)
$$H(k_x, y) = \sin k_x \sigma_x + \sin k_y \sigma_y + (m(y) + \cos k_x + \cos k_y)\sigma_z$$
A spatial domain wall in $m(y)$ induces chiral interface modes that bridge the bulk band gap.

### 🔹 Wilson–Dirac (2D)
$$H = \sum_i \sin k_i \gamma_i + \left(m(y) + r\sum_i (1 - \cos k_i)\right)\gamma_0$$
Implemented in a **strip geometry** (periodic in $x$, open in $y$), utilizing Wilson mass to suppress fermion doublers.

---

## ✅ Key Features

*   **Unified Topological Suite:** 1D Jackiw–Rebbi to 2D Wilson–Dirac support.
*   **Spectral Flow Analysis:** Automated tracking of eigenvalue evolution vs. momentum.
*   **Localization Diagnostics:** High-resolution spatial profiles of interface modes.
*   **Disorder Robustness Probing:** Evaluating topological stability under perturbations.
*   **Automated Visualization:** Built-in pipelines for spectrum, density maps, and spectral flow.
*   **XLA Optimized:** Seamless transition from CPU to GPU/TPU via JAX.

---

## 📁 Repository Structure

```text
topo_domain_wall/
├── topo_dw/
│   ├── __init__.py          # Core package initialization
│   ├── mass_profiles.py     # Functional mass distributions (Step, Kink, Tanh)
│   ├── hamiltonians.py      # Core operator construction (JIT-compatible)
│   ├── spectrum.py          # Parallelized spectral solvers
│   ├── interface.py         # Boundary mode and localization analysis
│   └── plotting.py          # Publication-quality visualization tools
├── run_demo_2d_wilson.py    # Primary orchestration script
├── README.md                # Project documentation
├── API.md                   # Functional API reference
└── LICENSE                  # MIT License
```

---

## 🚀 Usage

### 🔧 Installation
```bash
pip install jax jaxlib matplotlib
```

### ▶ Run 2D Wilson–Dirac Demo
Execute the primary diagnostic pipeline to generate topological profiles:
```bash
python run_demo_2d_wilson.py
```

**Output Assets:**
*   `wilson2d_spectrum.png`: Band structure showing crossing interface modes.
*   `wilson2d_interface_modes.png`: Spatial probability density of boundary states.
*   `wilson2d_spectral_flow.png`: Near-zero energy evolution across the Brillouin zone.

---

## ⚠ Pedagogical Design & Limitations

This framework is intentionally designed for **clarity and academic transparency**. Known constraints include:
*   **Model Scope:** Currently focuses on single-particle physics (no U(1)/SU(3) gauge fields or many-body interactions).
*   **Zero-Temperature:** Formalism is restricted to T=0 spectral analysis.
*   **Scaling:** Matrix diagonalization follows $O(N_y^3)$ scaling; extremely large systems may require custom sparse solvers.
*   **Optimization:** While JAX-accelerated, the Hamiltonian construction utilizes procedural loops for pedagogical readability.

---

## 📖 Citation

If this framework facilitates your research or educational projects, please cite it as follows:

```bibtex
@misc{hardiyan_topodwf_2026,
  author       = {Hari Hardiyan and Microsoft Copilot},
  title        = {Topological Domain-Wall Fermions Framework (JAX)},
  year         = {2026},
  howpublished = {\url{https://github.com/harihardiyan/topo_domain_wall}},
  note         = {AI Orchestration & Computational Physics}
}
```

---

## 👤 Author

**Hari Hardiyan**  
*AI Orchestration & Computational Physics ft. Microsoft Copilot*  
Specializing in hardware-accelerated topological research and automated numerical auditing.
