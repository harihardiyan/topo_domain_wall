
# Installation Guide

This document explains how to install and set up the  
**Topological Domain-Wall Fermions Framework (JAX)**.

---

## 1. Requirements

- Python 3.10 or newer
- pip (Python package manager)
- A working installation of JAX (CPU or GPU version)

The framework runs on:

- Linux
- macOS
- Windows (WSL recommended for GPU support)

---

## 2. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```
This installs:

jax

jaxlib

matplotlib

numpy

If you want GPU acceleration, install the appropriate JAX wheel from:

https://github.com/google/jax#installation (github.com in Bing)

3. Install the Framework Locally

From the project root directory:
```
pip install .
```
This installs the package as topo_dw, allowing you to import it:

```
import topo_dw
```
4. Optional: Developer Installation

If you plan to modify the source code:
```
pip install -e .
```
This enables live editing without reinstalling.

5. Testing the Installation

Run the included test suite:
```
pytest -q
```
If all tests pass, the installation is correct.

6. Running an Example

Try the Wilson–Dirac demo:
```
python run_demo_2d_wilson.py
```
This generates several PNG plots:

Spectrum

Interface modes

Spectral flow

7. GPU Support (Optional)

To enable GPU acceleration:

Install CUDA-compatible jaxlib

Ensure your GPU drivers are up to date

Verify JAX detects your GPU:
```
import jax
jax.devices()
```
8. Troubleshooting

JAX not detecting GPU

Install the correct CUDA wheel from the JAX documentation.

Matplotlib backend errors

Use a non-interactive backend:

```
import matplotlib
matplotlib.use("Agg")
```

Permission issues

Try:

```
pip install --user .
```
Installation is complete.You are now ready to explore topological fermions with JAX.

