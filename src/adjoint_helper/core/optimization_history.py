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

import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, computed_field

from adjoint_helper.core.defs import WeightsType
from adjoint_helper.utils.logging_utils import am_mpi_master

from ..core.base_settings import OptimizationSettings, SimulationSettings

logger = logging.getLogger(__name__)


class OptimizationHistory(BaseModel):
    optimization: OptimizationSettings

    @computed_field
    @property
    def fingerprint(self) -> str:
        return self.optimization.simulation_settings.get_fingerprint()

    @computed_field
    @property
    def settings(self) -> SimulationSettings:
        return self.optimization.simulation_settings

    def save_to_json(self) -> None:
        fpath = (
            self.optimization.simulation_settings.data_dir
            / self.optimization.simulation_settings.history_fname
        )
        if am_mpi_master():
            fpath.parent.mkdir(parents=True, exist_ok=True)
            data = self.model_dump()

            settings_dict = self.optimization.simulation_settings.model_dump()
            settings_dict["__class_path__"] = (
                self.optimization.simulation_settings.class_path
            )

            data["settings"] = settings_dict

            opt_dict = self.optimization.model_dump()
            opt_dict["__class_path__"] = self.optimization.class_path

            data["optimization"] = opt_dict

            def stringify_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
                if hasattr(obj, "tolist"):
                    return obj.tolist()
                if hasattr(obj, "item"):
                    return obj.item()
                if isinstance(obj, dict):
                    return {k: stringify_paths(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [stringify_paths(i) for i in obj]
                return obj

            data = stringify_paths(data)

            tmp_file = fpath.with_suffix(".tmp")
            try:
                with open(tmp_file, "w") as f:
                    json.dump(data, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno())

                tmp_file.replace(fpath)

                logger.info(f"History saved to {fpath}")

            except Exception as e:
                if tmp_file.exists():
                    tmp_file.unlink()
                logger.exception(f"Failed to save history: {e}")
                raise

    @classmethod
    def load_or_create(
        cls,
        optimization: OptimizationSettings,
        fpath: Path | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory":
        if fpath is None:
            fpath = optimization.simulation_settings.history_path

        hist = OptimizationHistory.load_if_exists(
            optimization=optimization,
            optimizer=optimization.optimizer,
            fpath=fpath,
            strict=strict,
        )

        if hist is None:
            hist = OptimizationHistory(optimization=optimization)

        return hist

    @classmethod
    def load_if_exists(
        cls,
        optimization: OptimizationSettings | None = None,
        optimizer: Any = None,
        fpath: Path | str | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory | None":
        """Because we cannot currently serialze and store the optimizer,
        you must supply it when loading history."""
        if fpath is None and optimization is not None:
            fpath = optimization.simulation_settings.history_path

        if isinstance(fpath, str):
            fpath = Path(fpath)

        if fpath is None:
            raise ValueError("No filepath supplied or located from the optimization")

        if fpath.is_file():
            logger.info(f"Loading History from {fpath}")
            return cls._load_from_json(
                optimization=optimization,
                optimizer=optimizer,
                fpath=fpath,
                strict=strict,
            )

        else:
            logger.info(f"No history found at {fpath}")
            return None

    @classmethod
    def _load_from_json(
        cls,
        optimization: OptimizationSettings | None = None,
        optimizer: Any = None,
        fpath: Path | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory | None":
        """Because we cannot currently serialze and store the optimizer,
        you must supply it when loading history."""
        if fpath is None and optimization is not None:
            fpath = optimization.simulation_settings.history_path

        if fpath is None:
            raise ValueError("No filepath supplied or located from the optimization")

        try:
            with open(fpath, "r") as f:
                data = json.load(f)

            # set_data = data["settings"]
            # if "__class_path__" in set_data:
            #     path_str = set_data.pop("__class_path__")
            #     mod_name, class_name = path_str.rsplit(".", 1)
            #     mod = importlib.import_module(mod_name)
            #     concrete_set_class = getattr(mod, class_name)
            #     settings_obj = concrete_set_class.model_validate(set_data)
            # else:
            #     settings_obj = SimulationSettings.model_validate(set_data)

            opt_data = data["optimization"]
            # TODO: Fix so we can store the optimizer so we don't
            # need to do this :(
            opt_data["optimizer"] = optimizer

            # In case we have a specialized SimulationSettings we need
            # to find it and load the appropriate module
            set_data = data["settings"]

            if "__class_path__" in set_data:
                path_str = set_data.pop("__class_path__")
                mod_name, class_name = path_str.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                concrete_set_class = getattr(mod, class_name)
                settings_obj = concrete_set_class.model_validate(set_data)
            else:
                raise ValueError(
                    "Something went wrong when loading the SimulationSettings -- you can"
                    " try to manually add missing __class_path__ key to the output file."
                )

            opt_data["simulation_settings"] = settings_obj

            if "__class_path__" in opt_data:
                path_str = opt_data.pop("__class_path__")
                mod_name, class_name = path_str.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                concrete_opt_class = getattr(mod, class_name)
                optimization_obj = concrete_opt_class.model_validate(opt_data)
            else:
                # TODO: Check if this (and settings below) always fail.
                # I believe they should, so this should probably be some kind
                # of error message
                raise ValueError(
                    "Something went wrong when loading the OptimizationSettings -- you can"
                    " try to manually add missing __class_path__ key to the output file."
                )
                optimization_obj = OptimizationSettings.model_validate(opt_data)

            hist = cls(optimization=optimization_obj)

            if optimization is not None:
                loaded_fingerprint = hist.fingerprint
                current_fingerprint = optimization.simulation_settings.get_fingerprint()

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
                        logger.warning(msg)

            return hist

        except Exception:
            logger.exception("Exception ocurred while loading history.")
            return None

    def optimize(self) -> list[WeightsType]:
        return self.optimization.optimize()
