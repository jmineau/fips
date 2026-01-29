import datetime as dt
from pathlib import Path

import pandas as pd

from stilt import Simulation


def get_sim(
    simulation: Simulation | Path,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None
) -> Simulation | str | None:
    sim = load_simulation(simulation)

    if sim.status != "SUCCESS":
        return sim.id

    if not sim_in_time_range(
        sim=sim, t_start=t_start, t_stop=t_stop, subset_hours=subset_hours
    ):
        return None
    return sim


def load_simulation(simulation: Simulation | Path) -> Simulation:
    if isinstance(simulation, Path):
        sim = Simulation.from_path(simulation)
    elif isinstance(simulation, Simulation):
        sim = simulation
    else:
        raise ValueError("simulation must be a Path or a stilt.Simulation object")
    return sim


def sim_in_time_range(
    sim: Simulation,
    t_start: dt.datetime,
    t_stop: dt.datetime,
    subset_hours: int | list[int] | None = None,
) -> bool:
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