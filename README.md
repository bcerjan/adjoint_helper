# adjoint_helper

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

`adjoint_helper` is a Python toolkit designed to simplify the implementation and execution of adjoint-based optimization workflows in electromagnetic simulations. It provides high-level abstractions for managing simulation parameters, calculating sensitivities, and interfacing with popular optimization engines.

## Features

* **Simplified Workflow:** Automate the boilerplate involved in adjoint sensitivity calculations.
* **Multi-Backend Ready:** Works with any program that returns gradients from your simulation (MEEP, Tidy3D, Legume, Elmer, ...)
* **Optimizer Integration:** Seamless connection to `nlopt` and `optax` for gradient-based optimization.
`AdjointDiffusion` as an optimization method is coming soon (hopefully)

## Installation

Via Pip (with Extras):

```bash
# Without any other solvers
pip install adjoint_helper

# With optax
pip install adjoint_helper[optax]

# With AdointDiffusion (not available yet)
pip install torch torchvision # possibly with --index-url=XXXXX, see below
pip install adjoint_helper[diffusion]
```

If you want to use [nlopt](https://nlopt.readthedocs.io/en/latest/) you should create a conda environment first and then use it to install pip so you can install adjoint_helper:

```bash
conda crate -n adjoint_helper
conda install -c conda-forge nlopt
conda install pip
pip install adjoint_helper
```

If you are running with `AdjointDiffusion` you also need to install pytorch
following the guidelines [here](https://pytorch.org/get-started/locally/) if
you want to use a GPU to accelerate it
---

## Quick Start

The first step is to define a custom `SimulationSettings` class and define
`create_geometry()` and `create_opt()` methods for it. From there, running an
optimization is as simple as:

```python
from your_custom_settings import YourCustomSettings
from adjoint_helper.optax_optimization import OptaxOptimizationSettings

settings = YourCustomSettings() # possibly with args here if needed
optimization = OptaxOptimizationSettings() # the defualts are sensible, uses `adam`

optimization.optimize(settings)
```

Clearly, this elides pretty much all the interesting pieces, so look in the `examples` directory for full examples that demonstrate the use of the package.
