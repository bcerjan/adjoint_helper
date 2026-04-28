from functools import singledispatch
from .base_settings import OptimizationSettings, SimulationSettings, PhysicsObjective


@singledispatch
def get_physics_objective(
    settings: SimulationSettings, optimization: OptimizationSettings
) -> PhysicsObjective:
    raise NotImplementedError(
        f"No physics objective registered for {type(optimization).__name__}."
    )
