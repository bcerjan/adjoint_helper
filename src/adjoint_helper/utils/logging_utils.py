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

import logging
import os
import sys

_LOGGING_INITIALIZED = False


def is_mpi_run() -> bool:
    mpi_env_vars = ["OMPI_COMM_WORLD_RANK", "PMI_RANK", "MPIRUN_RANK", "SLURM_PROCID"]
    return any(var in os.environ for var in mpi_env_vars)


def get_mpi_rank() -> int:
    if not is_mpi_run():
        return 0

    rank_str = (
        os.environ.get("OMPI_COMM_WORLD_RANK")
        or os.environ.get("PMI_RANK")
        or os.environ.get("SLURM_PROCID")
        or "0"
    )

    rank = int(rank_str)

    return rank


def am_mpi_master() -> bool:
    return get_mpi_rank() == 0


class MPIMasterFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return am_mpi_master()


class ColorFormatter(logging.Formatter):
    GREY = "\x1b[38;2;120;120;120m"
    GREEN = "\x1b[38;2;40;167;69m"
    YELLOW = "\x1b[38;2;255;193;7m"
    RED = "\x1b[38;2;220;53;69m"
    BOLD_RED = "\x1b[1;\x1b[38;2;220;53;69m"
    RESET = "\x1b[0m"

    LEVEL_COLORS = {
        logging.DEBUG: GREY,
        logging.INFO: GREEN,
        logging.WARNING: YELLOW,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        color = self.LEVEL_COLORS.get(record.levelno, self.RESET)

        orig_levelname = record.levelname

        record.levelname = f"{color}{orig_levelname}{self.RESET}"

        result = super().format(record)

        record.levelname = orig_levelname
        return result


def setup_logger(env_var_name: str = "ADJOINT_HELPER_LOG_LEVEL"):

    # Check if we already exist:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        _LOGGING_INITIALIZED = True
        return

    initial_level = os.environ.get(env_var_name, "INFO").upper()
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    level = valid_levels.get(initial_level, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(MPIMasterFilter())

    if sys.stdout.isatty():
        formatter = ColorFormatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
