from functools import singledispatch
from typing import Any
from .base_settings import OptimizationSettings, SimulationSettings
from .defs import PhysicsObjective


@singledispatch
def get_physics_objective(
    settings: SimulationSettings[Any], optimization: OptimizationSettings
) -> PhysicsObjective:
    raise NotImplementedError(
        f"No physics objective registered for {type(optimization).__name__}."
    )
