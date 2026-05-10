"""
Main entry point for Adjoint Helper
"""

"""
Adjoint Helper
Copyright (C) 2026 Ben Cerjan

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from .core.export_settings import (
    SingleRegionSettings,
    MultiRegionSettings,
    OptimizationSettings,
)

from .core.optimization_history import OptimizationHistory

from .core.defs import MaskRegion, Edge, PhysicsObjective, ObjectiveReturn


# from .optimizers.nlopt_optimization import NloptOptimizationSettings
# from .optimizers.optax_optimization import OptaxOptimizationSettings
# from .optimizers.nlopt_epigraph_optimization import NloptEpigraphOptimizationSettings
# from .optimizers.diffusion_optimization import AdjointDiffusionSettings # Not ready yet
