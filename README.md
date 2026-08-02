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

Clearly, this elides pretty much all the interesting pieces, so look in the `examples` directory for full examples that demonstrate the use of the package. Basically, you need to
write the code that defines your simulation and then the package can
take over from there.

### Important Notes for Save/Load

#### General

When saving the history it will overwrite any existing history with the
same name (without a warning), so rename things or change the filename if
you want to save the old results.

When loading and restarting, the optimization will start at the same sigmoid bias
it last completed, but at the first evaluation thereof (as a safe general assumption).
For example, if you had `sigmoid_biases = [4,16]` and `max_evals = [10, 20]` and 15
iterations had been completed, when restarting it would begin at `sigmoid_bias = 20` but
`iteration = 10` (that is, we finished all the `4` iterations and are now starting
on `16`). This also should help with regenerating the 'momentum' of the optimization
process as that gets lost during the save-load process.

To load history, there are currently 2 methods:

```python
OptimizationHistory.load_or_create() # HEAVILY preferred
OptimizationHistory.load_if_exists()
```

As noted in the comment, `load_or_create` is far and away the one that should be used
by default as it lets you just get on with things. If you are doing something
fancy, you _might_ want `load_if_exists` but I would think about why you're doing
it that way.

#### Settings

At present there isn't a method for storing the symmetry functions passed
to a `DesignRegion` so they must be re-applied after a load:

```python
hist = OptimizationHistory.load_or_create(settings=settings, default_optimization=optimization)

for dr in hist.settings.design_regions:
  dr.symmetry_func = lambda w: w/2 + npa.flipud(w/2)
```

#### Optimizer

Currently there is not a good way to store the state/configuration of the
optimizer, so you must re-pass it in when loading again. This is only
relevant when using `load_if_exists` as the optimizer from the provided
`optimization` will be used when using `load_or_create` (which is one
of the main reasons it is the preferred method). Note this means that
"momentum" will be lost when saving and re-loading. If you have a solution
for this, submit a PR!
