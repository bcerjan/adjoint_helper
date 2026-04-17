# adjoint_helper

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)

`adjoint_helper` is a Python toolkit designed to simplify the implementation and execution of adjoint-based optimization workflows in electromagnetic simulations. It provides high-level abstractions for managing simulation parameters, calculating sensitivities, and interfacing with popular optimization engines.

## Features

* **Simplified Workflow:** Automate the boilerplate involved in adjoint sensitivity calculations.
* **Multi-Backend Ready:** Designed to work with MEEP today, with support for other simulation engines coming soon.
* **Optimizer Integration:** Seamless connection to `nlopt` and `optax` for gradient-based optimization.
* **Parameter Management:** Easy handling of geometric and material parameters during optimization loops.

## Installation

Because this package relies on **MEEP** (at present), the recommended way to install `adjoint_helper` is via **Conda**.

### 1. Create a Conda Environment (Recommended)

First, create an environment with the necessary simulation backends:

```bash
# Create environment with MEEP and optimization tools
conda create -n adjoint_env -c conda-forge meep nlopt matplotlib numpy
conda activate adjoint_env
```
(nlopt is optional if you only want to use `optax` methods)

### 2. Install adjoint_helper
Once your environment is set up, you can install the package:

Via Pip (with Extras):
```bash
# Without any other solvers
pip install adjoint_helper

# With optax
pip install adjoint_helper[optax]
```

---
## Quick Start
The first step is to define a custom `SimulationSettings` class and define 
`create_geometry()` and `create_opt()` methods for it. From there, running an
optimization is as simple as:

```python
from your_custom_settings import YourCustomSettings
from adjoint_helper.optimization_settings import OptimizationSettings
from adjoint_helper.adam_optimization import run_adam_optimization

settings = YourCustomSettings() # possibly with args here if needed
optimization = OptimizationSettings() # the defualts work best for adam optimization

run_adam_optimization(settings, optimization)
```

There is an extended example in the `tests` directory that shows a more complicated
system and demonstrates how to subclass for your specific geometry.
