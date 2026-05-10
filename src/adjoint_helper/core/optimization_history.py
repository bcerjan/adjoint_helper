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

from pydantic import BaseModel, computed_field
from pathlib import Path

from ..core.base_settings import SimulationSettingsBase, OptimizationSettings


class OptimizationHistory(BaseModel):
    optimization: OptimizationSettings
    settings: SimulationSettingsBase

    @computed_field
    @property
    def fingerprint(self) -> str:
        return self.settings.get_fingerprint()

    def save_to_json(self, fpath: Path) -> None:
        with open(fpath, "w") as f:
            f.write(self.model_dump_json(indent=4))
        print(f"History saved to {fpath}")

    @classmethod
    def load_from_json(
        cls, fpath: Path, current_settings: SimulationSettingsBase, strict: bool = True
    ) -> "OptimizationHistory | None":

        try:
            with open(fpath, "r") as f:
                hist = cls.model_validate_json(f.read())

            loaded_fingerprint = hist.fingerprint
            current_fingerprint = current_settings.get_fingerprint()

            if loaded_fingerprint != current_fingerprint:
                msg = (
                    f"\nWARNING: FINGERPRINT MISMATCH!\n"
                    f"The loaded checkpoint was created for a DIFFERENT simulation.\n"
                    f"Loaded Fingerprint: {loaded_fingerprint[:8]}...\n"
                    f"Current Fingerprint: {current_fingerprint[:8]}...\n"
                    f"Proceeding may produce physically invalid results."
                )

                if strict:
                    raise ValueError(msg)
                else:
                    print(msg)

            return hist

        except Exception as e:
            print(f"Error loading history: {e}")
            return None
