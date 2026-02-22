"""Tests for fips aggregators module."""

import pandas as pd
import pytest

from fips.aggregators import integrate_over_time_bins


def test_integrate_over_time_bins_series():
    """Test integrating a Series over time bins."""
    # Create daily data
    idx = pd.date_range("2023-01-01", periods=10, freq="h", name="time")
    data = pd.Series(1.0, index=idx, name="flux")

    # Create 2-hour bins: 5 bins for 10 hours
    # Intervals: [00:00, 02:00), [02:00, 04:00), ...
    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=5, freq="2h", closed="left"
    )

    result = integrate_over_time_bins(data, bins, time_dim="time")

    assert len(result) == 5
    assert result.index.name == "time"
    # Each bin sums 2 hours of value 1.0 = 2.0
    assert (result.values == 2.0).all()


def test_integrate_over_time_bins_dataframe_multiindex():
    """Test integrating DataFrame with MultiIndex."""
    # 2 locations, 4 time points
    times = pd.date_range("2023-01-01", periods=4, freq="h")
    locs = ["A", "B"]
    idx = pd.MultiIndex.from_product([times, locs], names=["time", "loc"])
    df = pd.DataFrame({"value": 1.0}, index=idx)

    # Bin every 2 hours, closed='left' to match [start, end)
    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=2, freq="2h", closed="left"
    )

    result = integrate_over_time_bins(df, bins, time_dim="time")

    # Result should have 2 bins * 2 locs = 4 rows
    assert len(result) == 4
    # Check levels preserved and reordered
    assert set(result.index.names) == {"time", "loc"}

    # Sum should be 2.0 per bin per loc
    assert (result["value"] == 2.0).all()


def test_integrate_raises_missing_dim():
    """Test error when time_dim is missing."""
    data = pd.Series([1, 2], index=pd.Index([1, 2], name="other"))
    bins = pd.interval_range(start=0, periods=2, freq=1)

    with pytest.raises(ValueError, match="not found"):
        integrate_over_time_bins(data, bins, time_dim="time")
