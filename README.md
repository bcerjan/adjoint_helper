# adjoint_helper

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

`adjoint_helper` is a Python toolkit designed to simplify the implementation and execution of adjoint-based optimization workflows in electromagnetic simulations. It provides high-level abstractions for managing simulation parameters, calculating sensitivities, and interfacing with popular optimization engines.

## Features

* **Simplified Workflow:** Automate the boilerplate involved in adjoint sensitivity calculations.
* **Multi-Backend Ready:** Works with any program that returns gradients from your simulation (MEEP, Tidy3D, Legume, Elmer, ...)
* **Optimizer Integration:** Has several optimization strategies integrated, `optax`,
`nlopt`, or `structural attention`

## Installation

Via Pip (with Extras):

```bash
# Without any other solvers
pip install adjoint_helper

# With optax
pip install adjoint_helper[optax]

# With FDTDx as a simulation backend:
pip install adjoint_helper[fdtdx]
```

If you want to use [nlopt](https://nlopt.readthedocs.io/en/latest/) (or meep)
you should create a conda environment first and then use it to install pip so you
 can install adjoint_helper:

```bash
conda create -n adjoint_helper
conda install -c conda-forge nlopt
conda install pip
pip install adjoint_helper
```

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

Clearly, this elides pretty much all the interesting pieces, so look in the
`examples` directory for full examples that demonstrate the use of the package.
Basically, you need to write the code that defines your simulation and then the
package can take over from there.

### Notes on Simulation Return

The most important (aside from making sure your simulation is correct) thing
for using the library is the return type from `settings.create_opt`. It should
return a function that is an [`OptimizationFunction`](src/adjoint_helper/core/defs.py#L73).
This is a function that takes in a list of 2D weights (one per design region)
and returns a [`SimulationReturn`](/src/adjoint_helper/core/defs.py#L70) --
a tuple containing a `list[float]` of objective values and a
`list[list[list[1D Gradients]]]`. The list of gradients should be organized
like this:

```python
obj_vals, grads = opt(weights)
for design_region in grads:
  for obj_function in design_region:
    for parameter in obj_function:
      grad_1d = grads[design_region][obj_function][parameter]
```

For most simulation backends there should be a utility to normalize the return
so that you don't need to do this manually.

If there is not a utility to automatically normalize the return
from your simulator of choice (located in [normalization_functions.py](/src/adjoint_helper/util/normalization_functions.py)), please submit a PR adding one for it.

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
OptimizationHistory.load_or_create() # preferred if actually running the optimization
OptimizationHistory.load_if_exists() # generally only for data analysis after a run
```

As noted in the comment, `load_or_create` is the one that should be used
by default as it lets you just get on with the optimization.

If you are just analyzing your data, you probably want `load_if_exists` as it
is easier to work with in that situation. You just need to check if it returns
`None` (indicating no history was located)

#### Symmetry Functions

At present there isn't a method for storing the symmetry functions passed
to a `DesignRegion` so they must be re-applied after a load:

```python
import adjoint_helper as ah

hist = ah.OptimizationHistory.load_or_create(default_optimization=optimization)

for dr in hist.settings.design_regions:
  dr.symmetry_func = lambda w: w/2 + ah.anp.flipud(w/2)
```

Note the use of `ah.anp.xxx`. This is a wrapper that selects between autograd
and JAX arrays automatically depending on which is being used elsewhere in the
code. It is _heavily_ encouraged to use the `ah.anp` versions of `numpy` functions
so that things work no matter which simulation engine you are using.

#### If Using JAX (FDTDx)

In your simulation settings class (or somewhere) you need to tell `adjoint_helper` to
use the JAX backend for differentiation instead of autograd. To do this, you need
to call `adjoint_helper.anp.set_differentiation_lib("jax")`. Otherwise you will
see weird errors about arrays being incompatible.

#### Optimizer

Currently there is not a good way to store the state/configuration of the
optimizer, so you must re-pass it in when loading again. This is only
relevant when using `load_if_exists` as the optimizer from the provided
`optimization` will be used when using `load_or_create` (which is one
of the main reasons it is the preferred method). Note this means that
"momentum" will be lost when saving and re-loading. If you have a solution
for this, submit a PR!

#### Roadmap

Ideally I would like to add other backends (tidy3D, legume, etc.) but I don't use
them as much and haven't had the time to do so. Feel free to submit a PR adding
an example using one of them along with appropriate `normalization_xxx` functions.

I would also like to add Adjoint Diffusion as an optimization strategy eventually,
along with any other newer methods that are created.

#### LLM Disclosure

No LLM-generated code was used in this project. Occasional use of LLMs as
search engines was performed, but all the bugs are my own.
