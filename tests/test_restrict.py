"""
Tests for Problem.restrict and Vector.combine_first: part of the state held at its prior, and put back.

Ported from the downstream package's tests of its Jacobian coverage filter
(drop the cells the observations barely see, at every time, hold them at the
prior through the offset, and put the prior back in the reported posterior).
The inputs are the same synthetic ones; the cells to drop are chosen here
with :func:`fips.aggregate.sum`, as that filter chose them.
"""

import numpy as np
import pandas as pd
import pytest

import fips
from fips import (
    Covariance,
    Exponential,
    Independent,
    Kronecker,
    Operator,
    Problem,
    Space,
    Vector,
)
from fips.linalg import KroneckerOperator

TIMES = pd.DatetimeIndex(["2020-01-01", "2020-02-01"])
LATS = [40.5, 40.6]
LONS = [-112.0, -111.9]
WEAK = (40.5, -112.0)  # (lat, lon) of the cell the observations barely see
OBS_INDEX = pd.MultiIndex.from_arrays(
    [["wbb"] * 3, pd.to_datetime(["2020-01-05", "2020-01-20", "2020-02-05"])],
    names=["obs_location", "obs_time"],
)
PRIOR_COV = Kronecker(
    Exponential("time", "30D"), Exponential(["lat", "lon"], 20.0, metric="haversine")
)


def problem(bias: bool = False) -> Problem:
    """Prior 1..8 over (time, lat, lon); a unit Jacobian (columns (lon, lat, time)) except the weak cell (1e-6)."""
    index = pd.MultiIndex.from_product(
        [TIMES, LATS, LONS], names=["time", "lat", "lon"]
    )
    flux = pd.Series(np.arange(len(index), dtype=float) + 1.0, index=index)
    columns = pd.MultiIndex.from_product(
        [LONS, LATS, TIMES], names=["lon", "lat", "time"]
    )
    H = pd.DataFrame(1.0, index=OBS_INDEX, columns=columns)
    weak = (columns.get_level_values("lat") == WEAK[0]) & (
        columns.get_level_values("lon") == WEAK[1]
    )
    H.loc[:, weak] = 1e-6
    blocks = {("concentration", "flux"): H}
    priors = {"flux": flux}
    prior_cov = PRIOR_COV.with_std(0.5)
    if bias:
        bias_index = pd.Index(TIMES, name="time")
        Hb = pd.DataFrame([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], OBS_INDEX, bias_index)
        blocks[("concentration", "bias")] = Hb
        priors["bias"] = pd.Series(0.0, index=bias_index)
        prior_cov = Covariance.block_diag(
            flux=prior_cov, bias=Independent().with_std(0.1)
        )
    prior = Vector.from_blocks(priors, name="prior")
    obs = Vector.from_series(
        pd.Series([3.0, 3.1, 4.0], index=OBS_INDEX), block="concentration"
    )
    return Problem(
        prior=prior,
        prior_cov=prior_cov,
        obs=obs,
        obs_cov=Independent().with_std(0.2),
        forward=Operator.from_blocks(blocks),
        offset=Vector(np.full(3, 2.0), obs.space, name="background"),
    )


def cells(index: pd.Index) -> set:
    """(lat, lon) pairs of the flux labels of a state index."""
    if "block" in index.names:
        index = index[index.get_level_values("block") == "flux"]
    return set(
        zip(index.get_level_values("lat"), index.get_level_values("lon"), strict=True)
    )


def well_covered(p: Problem, percentile: float) -> Space:
    """
    The state without the cells whose coverage is below `percentile`, as the coverage filter chose them.

    A cell's coverage is the absolute sensitivity of every observation to it,
    summed over time, per observation location.
    """
    H = p.forward["concentration", "flux"]
    column_sums = Vector(np.abs(H.dense()).sum(axis=0), H.cols)
    per_cell = fips.aggregate.sum(column_sums.space, by=["lat", "lon"])
    locations = p.obs.index.get_level_values("obs_location").nunique()
    coverage = (per_cell @ column_sums).values / locations
    kept_cells = per_cell.rows.select("flux")[
        coverage >= np.percentile(coverage, percentile)
    ]
    flux = p.prior.space.select("flux")
    is_kept = pd.MultiIndex.from_arrays(
        [flux.get_level_values("lat"), flux.get_level_values("lon")]
    ).isin(kept_cells)
    blocks = {"flux": flux[is_kept]}
    if "bias" in p.prior.space:
        blocks["bias"] = p.prior.space.select("bias")
    return Space(blocks)


def test_weak_cell_leaves_prior_forward_and_prior_cov():
    """The weak cell goes at every time, from the prior, the forward operator and the prior covariance."""
    p = problem()
    keep = well_covered(p, 30)
    small = p.restrict(keep)

    retained = cells(p.prior.index) - {WEAK}
    assert cells(keep.index) == retained
    assert cells(small.prior.index) == retained
    assert cells(small.forward.cols.index) == retained
    assert cells(small.prior_cov.space.index) == retained
    assert len(small.prior["flux"]) == len(retained) * len(TIMES)
    # the kept cells keep their prior values
    kept = small.prior.to_series()
    full = p.prior.to_series()
    np.testing.assert_array_equal(kept.to_numpy(), full.reindex(kept.index).to_numpy())


def test_weak_cell_is_held_at_the_prior_in_the_offset():
    """The weak cell's prior (1 in January, 5 in February), seen with 1e-6 by every observation, joins the offset."""
    small = problem().restrict(well_covered(problem(), 30))
    np.testing.assert_allclose(
        small.offset.values, 2.0 + 1e-6 * (1.0 + 5.0), rtol=1e-15
    )
    assert small.offset.name == "background"


def test_the_contribution_is_added_by_label():
    """A forward operator whose rows are in another order than the observations is added to the offset by label."""
    p = problem()
    backwards = Space.from_index(p.obs.index[::-1])
    moved = Problem(
        prior=p.prior,
        prior_cov=p.prior_cov,
        obs=p.obs,
        obs_cov=p.obs_cov,
        forward=p.forward.align(backwards, p.forward.cols),
        offset=Vector(np.array([2.0, 2.5, 3.0]), p.obs.space),
    )
    x = p.prior.values
    H = p.forward.dense()
    is_weak = np.array(
        [
            cell == WEAK
            for cell in zip(
                p.prior.index.get_level_values("lat"),
                p.prior.index.get_level_values("lon"),
                strict=True,
            )
        ]
    )
    expected = np.array([2.0, 2.5, 3.0]) + H[:, is_weak] @ x[is_weak]
    np.testing.assert_allclose(
        moved.restrict(well_covered(p, 30)).offset.values, expected
    )


def test_keeping_everything_changes_nothing():
    """A threshold below every cell keeps the problem as it was."""
    p = problem()
    same = p.restrict(well_covered(p, 0))
    assert same.prior.space == p.prior.space
    np.testing.assert_array_equal(same.offset.values, p.offset.values)
    np.testing.assert_array_equal(same.prior_cov.dense(), p.prior_cov.dense())


def test_without_an_offset_the_contribution_is_the_offset():
    """A problem with no offset gets one: the cells left out."""
    p = problem()
    bare = Problem(
        prior=p.prior,
        prior_cov=p.prior_cov,
        obs=p.obs,
        obs_cov=p.obs_cov,
        forward=p.forward,
    )
    small = bare.restrict(well_covered(p, 30))
    np.testing.assert_allclose(small.offset.values, 1e-6 * (1.0 + 5.0), rtol=1e-15)
    assert bare.restrict(bare.prior.space).offset is None


def test_bias_block_is_kept():
    """A block named in keep stays whole: its prior, its columns and its covariance."""
    p = problem(bias=True)
    small = p.restrict(well_covered(p, 30))
    np.testing.assert_array_equal(small.prior["bias"].values, p.prior["bias"].values)
    assert small.forward.cols.names == ("flux", "bias")
    assert len(small.forward.cols.select("bias")) == len(TIMES)
    np.testing.assert_allclose(small.prior_cov["bias", "bias"].variances.values, 0.1**2)
    assert cells(small.prior_cov.space.index) == cells(small.prior.index)


def test_a_block_left_out_is_held_at_the_prior():
    """Keep without the bias block holds every offset at its prior (zero here)."""
    p = problem(bias=True)
    flux_only = Space({"flux": p.prior.space.select("flux")})
    small = p.restrict(flux_only)
    assert small.prior.space.names == ("flux",)
    np.testing.assert_array_equal(small.offset.values, p.offset.values)


def test_prior_cov_keeps_its_form_and_is_the_model_on_the_kept_cells():
    """The Kronecker factors stay factors, and the submatrix is the model built on the kept labels."""
    p = problem()
    keep = well_covered(p, 30)
    small = p.restrict(keep)
    assert isinstance(p.prior_cov.data, KroneckerOperator)
    assert isinstance(small.prior_cov.data, KroneckerOperator)
    rebuilt = PRIOR_COV.with_std(0.5).build(keep)
    np.testing.assert_allclose(
        small.prior_cov.dense(), rebuilt.dense(), rtol=1e-15, atol=0
    )


def test_restricted_solve_is_the_problem_with_the_cells_fixed():
    """The estimate is the closed form of the smaller problem, with the cells left out added to the offset."""
    p = problem(bias=True)
    keep = well_covered(p, 30)
    post = fips.solve(p.restrict(keep))
    k = p.prior.space._subset(keep, "keep")
    r = np.setdiff1d(np.arange(len(p.prior)), k)
    H, x0, S0 = p.forward.dense(), p.prior.values, p.prior_cov.dense()
    S_z = p.obs_cov.dense()
    c = p.offset.values + H[:, r] @ x0[r]
    Hk, S0k = H[:, k], S0[np.ix_(k, k)]
    G = Hk @ S0k @ Hk.T + S_z
    expected = x0[k] + S0k @ Hk.T @ np.linalg.solve(G, p.obs.values - Hk @ x0[k] - c)
    np.testing.assert_allclose(post.estimate.values, expected, rtol=1e-10)


def test_reconstruct_puts_the_prior_back_in_removed_cells():
    """The estimate on the whole state: the solved cells, and the prior where the state was held."""
    p = problem()
    small = p.restrict(well_covered(p, 30))
    posterior = Vector(small.prior.values * 10, small.prior.space, name="posterior")

    full = posterior.combine_first(p.prior)

    assert full.name == "posterior"
    assert full.space == p.prior.space
    index = full.index
    is_weak = (index.get_level_values("lat") == WEAK[0]) & (
        index.get_level_values("lon") == WEAK[1]
    )
    np.testing.assert_array_equal(full.values[is_weak], p.prior.values[is_weak])
    np.testing.assert_array_equal(full.values[~is_weak], p.prior.values[~is_weak] * 10)


def test_keep_must_be_part_of_the_state():
    """A block or a label the prior lacks is named; keep is a Space."""
    p = problem()
    with pytest.raises(ValueError, match=r"blocks not in the space: 'bias'"):
        p.restrict(Space({"bias": pd.Index(TIMES, name="time")}))
    flux = p.prior.space.select("flux")
    other = flux.set_levels(
        pd.DatetimeIndex(["2020-03-01", "2020-04-01"]), level="time"
    )
    with pytest.raises(ValueError, match=r"labels of block 'flux' not in the space"):
        p.restrict(Space({"flux": other}))
    with pytest.raises(ValueError, match="has levels"):
        p.restrict(Space({"flux": pd.Index([1], name="cell")}))
    with pytest.raises(TypeError, match="keep must be a Space"):
        p.restrict(flux)


def test_a_subclass_stays_a_subclass():
    """Restrict and aggregate give a problem of the class they are called on."""

    class MyProblem(Problem):
        pass

    p = problem()
    mine = MyProblem(
        prior=p.prior,
        prior_cov=p.prior_cov,
        obs=p.obs,
        obs_cov=p.obs_cov,
        forward=p.forward,
    )
    assert type(mine.restrict(well_covered(p, 30))) is MyProblem
    W = fips.aggregate.mean(p.obs.space, by="obs_location")
    assert type(mine.aggregate(W)) is MyProblem
