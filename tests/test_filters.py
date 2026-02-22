"""Tests for fips filters module."""

import pandas as pd

from fips.filters import enough_obs_per_interval


def test_enough_obs_per_interval_simple():
    """Test basic threshold filtering."""
    # 3 obs in bin 1, 1 obs in bin 2
    times = pd.to_datetime(
        ["2023-01-01 00:00", "2023-01-01 00:30", "2023-01-01 00:45", "2023-01-01 01:15"]
    )
    idx = pd.Index(times, name="time")

    # Bins: [00:00, 01:00), [01:00, 02:00). Use closed='left'
    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=2, freq="1h", closed="left"
    )

    # Threshold=2: First 3 obs (bin 1) pass, last 1 obs (bin 2) fails
    mask = enough_obs_per_interval(idx, bins, threshold=2)

    assert len(mask) == 4
    assert mask == [True, True, True, False]


def test_enough_obs_per_interval_multiindex():
    """Test filtering with MultiIndex level."""
    # (time, id)
    times = pd.to_datetime(["2023-01-01 00:00", "2023-01-01 01:00"])
    idx = pd.MultiIndex.from_arrays([times, [1, 2]], names=["time", "id"])

    bins = pd.interval_range(
        start=pd.Timestamp("2023-01-01 00:00"), periods=2, freq="1h", closed="left"
    )

    # 1 obs per bin. Threshold 2 -> all False
    mask = enough_obs_per_interval(idx, bins, threshold=2, level="time")
    assert mask == [False, False]

    # Threshold 1 -> all True
    mask = enough_obs_per_interval(idx, bins, threshold=1, level="time")
    assert mask == [True, True]
