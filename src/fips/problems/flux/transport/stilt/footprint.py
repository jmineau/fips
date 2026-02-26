import datetime as dt
import logging

from stilt import Footprint, Simulation  # type: ignore[import]

logger = logging.getLogger(__name__)


def get_footprint(
    sim: Simulation,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    resolution: str | None = None,
) -> Footprint | None:
    # Load footprint object
    footprint = load_footprint(sim=sim, resolution=resolution)

    if not footprint:
        logger.debug(f"No footprint available for simulation {sim.id}")
        return None

    # Check if footprint time range overlaps with inversion time range
    if not footprint_in_time_range(footprint=footprint, t_start=t_start, t_stop=t_stop):
        # A footprint within a simulation might not overlap with the inversion time range
        # even if the simulation does
        logger.debug(f"Footprint for simulation {sim.id} outside inversion time range")
        return None

    return footprint


def load_footprint(sim: Simulation, resolution: str | None = None) -> Footprint | None:
    # Load footprint at specified resolution if available
    # Otherwise, get the default (highest) resolution footprint
    footprint = sim.footprints[resolution] if resolution else sim.footprint
    logger.debug(f"Loaded footprint for simulation {sim.id} (resolution={resolution})")
    return footprint


def footprint_in_time_range(
    footprint: Footprint, t_start: dt.datetime, t_stop: dt.datetime
) -> bool:
    # Check if footprint time range overlaps with inversion time range
    # A footprint within a simulation might not overlap with the inversion time range
    # even if the simulation does
    return footprint.time_range[0] < t_stop and footprint.time_range[1] > t_start
