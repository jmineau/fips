"""
Tests for fips.scales: the sizes of each observation's signal, for a model-data mismatch that grows with it.

Ported from the downstream package's test of its multiplicative mismatch
scale (``scale_on="prior"`` matched by label), with the other two scales
(``"footprint"`` and ``"obs"``) written out the same way.
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

import fips
from fips import Independent, Operator, Space, Vector

TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * 3, pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-05"])],
    names=["obs_location", "obs_time"],
)


def flux_prior() -> tuple[pd.Series, Vector]:
    """A prior ordered (time, lat, lon), like an inventory's."""
    index = pd.MultiIndex.from_product(
        [TIMES, LATS, LONS], names=["time", "lat", "lon"]
    )
    series = pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index)
    return series, Vector.from_series(series, block="flux")


def jacobian(sparse: bool = False) -> tuple[pd.DataFrame, Operator]:
    """A forward operator whose columns are ordered (lon, lat, time), unlike the prior."""
    columns = pd.MultiIndex.from_product(
        [LONS, LATS, TIMES], names=["lon", "lat", "time"]
    )
    rng = np.random.default_rng(0)
    values = rng.random((len(OBS_INDEX), len(columns))) - 0.3
    H = pd.DataFrame(values, OBS_INDEX, columns)
    op = Operator.from_frame(H, row_block="concentration", col_block="flux")
    if sparse:
        op = Operator(sp.csr_array(op.dense()), op.rows, op.cols)
    return H, op


def test_modeled_matches_by_label():
    """|H x_0| with the prior matched to the Jacobian's columns by label, not position."""
    prior_s, prior = flux_prior()
    H, forward = jacobian()
    x = prior_s.reorder_levels(H.columns.names).reindex(H.columns)
    expected = np.abs(H.to_numpy() @ x.to_numpy())
    positional = np.abs(H.to_numpy() @ prior_s.to_numpy())
    assert not np.allclose(expected, positional)  # the orders really differ here

    scale = fips.scales.modeled(forward, prior)
    assert isinstance(scale, Vector) and scale.name == "modeled"
    assert scale.space == forward.rows
    np.testing.assert_allclose(scale.values, expected, rtol=1e-14)


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_footprint(sparse):
    """Each row's absolute sum times the prior's mean, dense or sparse alike."""
    prior_s, prior = flux_prior()
    H, forward = jacobian(sparse)
    expected = np.abs(H.to_numpy()).sum(axis=1) * float(np.mean(prior_s.to_numpy()))
    scale = fips.scales.footprint(forward, prior)
    assert scale.name == "footprint" and scale.space == forward.rows
    np.testing.assert_allclose(scale.values, expected, rtol=1e-14)


def test_observed():
    """|z - c|, with the offset matched by label, a number, or none."""
    z = Vector.from_series(
        pd.Series([2.0, 2.2, 1.7], index=OBS_INDEX), block="concentration"
    )
    background = Vector.from_series(
        pd.Series([1.9, 2.0, 1.9], index=OBS_INDEX).iloc[::-1], block="concentration"
    )
    np.testing.assert_allclose(
        fips.scales.observed(z, background).values, [0.1, 0.2, 0.2]
    )
    np.testing.assert_allclose(fips.scales.observed(z, 2.0).values, [0.0, 0.2, 0.3])
    np.testing.assert_array_equal(fips.scales.observed(z).values, z.values)
    assert fips.scales.observed(z).name == "observed"


def test_a_scale_makes_a_mismatch():
    """A fraction of a scale is a standard deviation for a model."""
    _, prior = flux_prior()
    _, forward = jacobian()
    scale = fips.scales.footprint(forward, prior)
    S_z = Independent().with_std(Vector(0.3 * scale.values, scale.space))
    assert S_z.space == forward.rows
    np.testing.assert_allclose(S_z.variances.values, (0.3 * scale.values) ** 2)


def test_other_labels_and_types_raise():
    """The prior must be on the Jacobian's columns; the inputs fips objects."""
    _, forward = jacobian()
    other = Vector(np.ones(2), Space({"flux": pd.Index(["a", "b"], name="cell")}))
    with pytest.raises(ValueError):
        fips.scales.footprint(forward, other)
    with pytest.raises(ValueError):
        fips.scales.modeled(forward, other)
    with pytest.raises(TypeError, match="forward must be an Operator"):
        fips.scales.modeled(forward.dense(), other)
    with pytest.raises(TypeError, match="prior must be a Vector"):
        fips.scales.footprint(forward, other.values)
    with pytest.raises(TypeError, match="obs must be a Vector"):
        fips.scales.observed(other.values)
