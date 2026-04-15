"""
Meep Adjoint Helper
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

import pickle
from adjoint_helper.simulation_settings import SimulationSettings
from adjoint_helper.optimization_settings import OptimizationSettings


class OptimizationHistory:
    optimization: OptimizationSettings
    settings: SimulationSettings

    def __init__(
        self, settings: SimulationSettings, optimization: OptimizationSettings
    ):
        self.settings = settings
        self.optimization = optimization

    def save_history(self, fname: str) -> None:
        """Convenience function to save the history to file. Will overwrite."""
        with open(fname, "wb") as file:
            pickle.dump(self, file)


def load_history(fname: str) -> OptimizationHistory:
    """Convenience method to load the history from file."""
    with open(fname, "rb") as file:
        history = pickle.load(file)
    return history
