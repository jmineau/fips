"""
Transport error estimation for STILT simulations.

This module provides functions for calculating transport errors from
STILT particle trajectory ensembles using regular and perturbed (error)
particle runs.
"""

from pathlib import Path

import pandas as pd
import xarray as xr


def extract_flux(fluxes, particles):
    """
    Extracts flux values from a flux dataset for given particle locations.

    Parameters
    ----------
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.
    particles : pandas.DataFrame
        DataFrame containing particle trajectory data with 'lati' and 'long' columns.

    Returns
    -------
    np.ndarray
        Numpy array of flux values corresponding to particle locations.
    """
    lat_indexer = xr.DataArray(particles.lati.values, dims="point")
    lon_indexer = xr.DataArray(particles.long.values, dims="point")
    selected_flux = fluxes.sel(lat=lat_indexer, lon=lon_indexer, method="nearest")
    return selected_flux.values


def calculate_particle_concentrations(
    trajectory_path: str | Path, fluxes: xr.DataArray
) -> pd.DataFrame:
    """
    Calculates the concentration of dCH4 for particles in a given trajectory file.

    Parameters
    ----------
    trajectory_path : str or Path
        Path to the trajectory parquet file.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    pd.DataFrame
        Concentration of dCH4 for the particles.
    """
    trajectory_path = Path(trajectory_path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    particles = pd.read_parquet(trajectory_path)

    particles["flux"] = extract_flux(fluxes=fluxes, particles=particles)
    particles["dCH4"] = particles["foot"] * particles["flux"]

    return particles.groupby("indx")["dCH4"].sum()


def calculate_particle_variance(
    trajectory_path: str | Path, fluxes: xr.DataArray
) -> float:
    """
    Calculates the variance of dCH4 for particles in a given trajectory file.

    Parameters
    ----------
    trajectory_path : str or Path
        Path to the trajectory parquet file.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    float
        Variance of dCH4 for the particles.
    """
    particles = calculate_particle_concentrations(trajectory_path, fluxes=fluxes)
    return float(particles.var())


def _resolve_paths(sim_dir: str | Path) -> tuple[Path, Path]:
    """Resolve regular and error trajectory paths from a simulation directory."""
    sim_dir = Path(sim_dir)
    sim_id = sim_dir.name
    return (
        sim_dir / f"{sim_id}_traj.parquet",
        sim_dir / f"{sim_id}_error.parquet",
    )


def plot_particle_variances(
    sim_dir: str | Path, fluxes: xr.DataArray
) -> "matplotlib.axes.Axes":
    """
    Plots the variances of dCH4 for regular and error particles.

    Parameters
    ----------
    sim_dir : str or Path
        Path to the simulation directory.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    plt.Axes
        Matplotlib Axes object with the plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    regular_path, error_path = _resolve_paths(sim_dir)

    regular = calculate_particle_concentrations(regular_path, fluxes=fluxes)
    error = calculate_particle_concentrations(error_path, fluxes=fluxes)

    df = pd.DataFrame({"Regular Particles": regular, "Error Particles": error}).melt(
        var_name="Particle Type", value_name="CH4 Concentration"
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(
        df, ax=ax, x="CH4 Concentration", hue="Particle Type", bins=100, kde=True
    )

    return ax


def calculate_transport_error(
    sim_dir: str | Path, fluxes: xr.DataArray
) -> float:
    """
    Calculates the transport error for a given STILT simulation directory.

    Parameters
    ----------
    sim_dir : str or Path
        Path to the simulation directory.
    fluxes : xarray.DataArray
        The flux dataset with 'lat' and 'lon' dimensions.

    Returns
    -------
    float
        The calculated transport error (error variance minus regular variance).
    """
    regular_path, error_path = _resolve_paths(sim_dir)
    return calculate_particle_variance(error_path, fluxes=fluxes) - \
        calculate_particle_variance(regular_path, fluxes=fluxes)
