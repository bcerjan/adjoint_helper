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
conda create -n adjoint_helper
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

### Important Notes for Save/Load

#### General

When loading and restarting, the optimization will start at the same sigmoid bias
it last completed, but at the first evaluation thereof (as a safe general assumption).
For example, if you had `sigmoid_biases = [4,16]` and `max_evals = [10, 20]` and 15
iterations had been completed, when restarting it would begin at `sigmoid_bias = 20` but
`iteration = 10` (that is, we finished all the `4` iterations and are now starting
on `16`)

#### Optimizer

Currently there is not a good way to store the state/configuration of the
optimizer, so you must re-pass it in when loading again.

#### Settings

At present there isn't a method for storing the symmetry functions passed
to a `DesignRegion` so they must be re-applied after a load:

```python
hist = OptimizationHistory.load_if_exists(settings=settings, optimizer=optimizer)
if hist is not None:
  for dr in hist.settings.design_regions:
    dr.symmetry_func = lambda w: w/2 + npa.flipud(w/2)
```
