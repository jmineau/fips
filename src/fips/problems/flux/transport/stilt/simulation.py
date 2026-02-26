"""
STILT simulation management.

This module provides functions for loading and managing STILT simulations,
including filtering by time and status, and extracting relevant metadata.
"""

import datetime as dt
import logging
from pathlib import Path

import pandas as pd
from stilt import Simulation  # type: ignore[import]

logger = logging.getLogger(__name__)


def get_sim(
    simulation: Simulation | Path,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None,
) -> Simulation | str | None:
    """
    Get simulation if it's successful and overlaps with time range.

    Parameters
    ----------
    simulation : Simulation or Path
        STILT simulation object or path to simulation directory.
    t_start : datetime
        Start of the inversion time range.
    t_stop : datetime
        End of the inversion time range.
    subset_hours : int or list[int], optional
        Hour(s) of day to filter simulations. If None, uses all hours.

    Returns
    -------
    Simulation or str or None
        Simulation object if successful and in time range, simulation ID string if failed, or None if outside time range.
    """
    sim = load_simulation(simulation)

    if sim.status != "SUCCESS":
        logger.debug(f"Skipping simulation {sim.id} with status {sim.status}")
        return str(sim.id)

    if not sim_in_time_range(
        sim=sim, t_start=t_start, t_stop=t_stop, subset_hours=subset_hours
    ):
        logger.debug(f"Simulation {sim.id} outside time range")
        return None
    return sim


def load_simulation(simulation: Simulation | Path) -> Simulation:
    """
    Load simulation from path or return existing Simulation object.

    Parameters
    ----------
    simulation : Simulation or Path
        STILT simulation object or path to simulation directory.

    Returns
    -------
    Simulation
        STILT simulation object.
    """
    if isinstance(simulation, Path):
        sim = Simulation.from_path(simulation)
    elif isinstance(simulation, Simulation):
        sim = simulation
    else:
        raise ValueError("simulation must be a Path or a stilt.Simulation object")
    logger.debug(f"Loaded simulation {sim.id}")
    return sim


def sim_in_time_range(
    sim: Simulation,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None,
) -> bool:
    """
    Check if simulation overlaps with inversion time range.

    Parameters
    ----------
    sim : Simulation
        STILT simulation object.
    t_start : datetime
        Start of the inversion time range.
    t_stop : datetime
        End of the inversion time range.
    subset_hours : int or list[int], optional
        Hour(s) of day to filter simulations. If None, uses all hours.

    Returns
    -------
    bool
        True if simulation overlaps with inversion time range, False otherwise.
    """
    n_hours = sim.config.n_hours
    if n_hours >= 0:
        raise ValueError("STILT must be run backwards in time (n_hours < 0)")
    n_hours = pd.to_timedelta(n_hours, unit="h")

    # Skip simulations that do not overlap with inversion time range
    sim_start = sim.receptor.time + n_hours
    sim_end = sim.receptor.time - pd.Timedelta(hours=1)
    if sim_end < t_start or sim_start >= t_stop:
        return False

    if subset_hours:
        # Subset simulations to specific hours
        if isinstance(subset_hours, int):
            subset_hours = [subset_hours]

        if sim.receptor.time.hour not in subset_hours:
            return False

    return True
