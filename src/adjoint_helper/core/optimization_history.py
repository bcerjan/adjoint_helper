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
import os
from typing import Any
from pydantic import BaseModel, computed_field
from pathlib import Path
import logging
import json

from adjoint_helper.core.defs import WeightsType
from adjoint_helper.utils.logging_utils import am_mpi_master

from ..core.base_settings import SimulationSettings, OptimizationSettings

logger = logging.getLogger(__name__)


class OptimizationHistory(BaseModel):
    optimization: OptimizationSettings
    settings: SimulationSettings

    @computed_field
    @property
    def fingerprint(self) -> str:
        return self.settings.get_fingerprint()

    def save_to_json(self) -> None:
        fpath = self.settings.data_dir / self.settings.history_fname
        if am_mpi_master():
            logger.info("Starting pydantic")
            fpath.parent.mkdir(parents=True, exist_ok=True)
            w = self.settings.weights.copy()
            self.settings.update_weights(w)
            logger.info("Updated weights")
            data = self.model_dump()
            logger.info("Dumped Model")

            settings_dict = self.settings.model_dump()
            logger.info("Dumped Settings")
            settings_dict["__class_path__"] = self.settings.class_path

            data["settings"] = settings_dict

            opt_dict = self.optimization.model_dump()
            logger.info("Dumped Optimization")
            opt_dict["__class_path__"] = self.optimization.class_path

            data["optimization"] = opt_dict

            def stringify_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
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
        settings: SimulationSettings,
        default_optimization: OptimizationSettings,
        fpath: Path | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory":
        if fpath is None:
            fpath = settings.history_path

        hist = OptimizationHistory.load_if_exists(
            settings=settings,
            optimizer=default_optimization.optimizer,
            fpath=fpath,
            strict=strict,
        )

        if hist is None:
            hist = OptimizationHistory(
                optimization=default_optimization, settings=settings
            )

        return hist

    @classmethod
    def load_if_exists(
        cls,
        settings: SimulationSettings,
        optimizer: Any,
        fpath: Path | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory | None":
        """Because we cannot currently serialze and store the optimizer,
        you must supply it when loading history."""
        if fpath is None:
            fpath = settings.history_path

        if fpath.is_file():
            logger.info(f"Loading History from {fpath}")
            return cls._load_from_json(
                settings=settings,
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
        settings: SimulationSettings,
        optimizer: Any,
        fpath: Path | None = None,
        strict: bool = True,
    ) -> "OptimizationHistory | None":
        """Because we cannot currently serialze and store the optimizer,
        you must supply it when loading history."""
        if fpath is None:
            fpath = settings.history_path

        try:
            with open(fpath, "r") as f:
                data = json.load(f)

            opt_data = data["optimization"]
            # TODO: Fix so we can store the optimizer so we don't
            # need to do this :(
            opt_data["optimizer"] = optimizer
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
                optimization_obj = OptimizationSettings.model_validate(opt_data)

            set_data = data["settings"]
            if "__class_path__" in set_data:
                path_str = set_data.pop("__class_path__")
                mod_name, class_name = path_str.rsplit(".", 1)
                mod = importlib.import_module(mod_name)
                concrete_set_class = getattr(mod, class_name)
                settings_obj = concrete_set_class.model_validate(set_data)
            else:
                settings_obj = SimulationSettings.model_validate(set_data)

            hist = cls(optimization=optimization_obj, settings=settings_obj)

            loaded_fingerprint = hist.fingerprint
            current_fingerprint = settings.get_fingerprint()

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
        return self.optimization.optimize(self.settings)
