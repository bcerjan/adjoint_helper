from functools import singledispatch
from .base_settings import OptimizationSettings, SimulationSettingsBase
from .defs import PhysicsObjective


@singledispatch
def get_physics_objective(
    settings: SimulationSettingsBase, optimization: OptimizationSettings
) -> PhysicsObjective:
    raise NotImplementedError(
        f"No physics objective registered for {type(optimization).__name__}."
    )
