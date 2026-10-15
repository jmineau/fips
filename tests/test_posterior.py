"""
Tests for the Posterior fips.solve returns.

Labels, types, names, caching, the diagnostics, the leverage and pickling.

The numbers are held to the closed form in ``test_oracle.py``; the
factorization (variances, aggregations, ``with_gamma``, ``with_data``,
memory) in ``test_whitened.py``.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

import fips
from fips import Covariance, Operator, Posterior, Problem, Space, Vector


def _spd(rng, n):
    a = rng.normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _problem(seed=0) -> Problem:
    """A two-block state (flux, bias), one obs block, and a background."""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2023-01", periods=3, freq="MS")
    flux_idx = pd.MultiIndex.from_product([times, ["A", "B"]], names=["time", "loc"])
    bias_idx = pd.Index(["S1", "S2"], name="site")
    obs_idx = pd.MultiIndex.from_tuples(
        [
            (t, s)
            for t in pd.date_range("2023-01-01", periods=4, freq="20D")
            for s in ["S1", "S2"]
        ],
        names=["time", "site"],
    )
    prior = Vector.from_blocks(
        {
            "flux": pd.Series(rng.normal(size=6), index=flux_idx),
            "bias": pd.Series([0.1, -0.1], index=bias_idx),
        },
        name="prior",
    )
    obs = Vector.from_series(
        pd.Series(400 + rng.normal(size=8), index=obs_idx, name="concentration"),
        block="concentration",
    )
    H = Operator.from_blocks(
        {
            ("concentration", "flux"): pd.DataFrame(
                rng.random((8, 6)), index=obs_idx, columns=flux_idx
            ),
            ("concentration", "bias"): pd.DataFrame(
                np.tile(np.eye(2), (4, 1)), index=obs_idx, columns=bias_idx
            ),
        }
    )
    S_0 = Covariance.from_blocks(
        {
            ("flux", "flux"): pd.DataFrame(
                _spd(rng, 6), index=flux_idx, columns=flux_idx
            ),
            ("bias", "bias"): Covariance.diagonal(0.01, Space({"bias": bias_idx})),
        }
    )
    S_z = Covariance.from_frame(
        pd.DataFrame(_spd(rng, 8), index=obs_idx, columns=obs_idx),
        row_block="concentration",
        col_block="concentration",
    )
    background = Vector(np.full(8, 399.0), obs.space, name="background")
    return Problem(
        prior=prior, prior_cov=S_0, obs=obs, obs_cov=S_z, forward=H, offset=background
    )


@pytest.fixture
def problem():
    """A two-block problem."""
    return _problem()


@pytest.fixture
def post(problem):
    """The posterior of ``problem``."""
    return fips.solve(problem)


# name, how to read it, the space of its rows ("state" or "obs") and of its
# columns (None for a vector), its class, its name
OUTPUTS = [
    ("estimate", lambda p: p.estimate, "state", None, Vector, "estimate"),
    ("variances", lambda p: p.variances, "state", None, Vector, "variance"),
    ("cov()", lambda p: p.cov(), "state", "state", Covariance, "posterior_cov"),
    ("kalman_gain", lambda p: p.kalman_gain, "state", "obs", Operator, "kalman_gain"),
    (
        "averaging_kernel()",
        lambda p: p.averaging_kernel(),
        "state",
        "state",
        Operator,
        "averaging_kernel",
    ),
    ("modeled_obs", lambda p: p.modeled_obs, "obs", None, Vector, "modeled_obs"),
    (
        "innovation_cov",
        lambda p: p.innovation_cov,
        "obs",
        "obs",
        Covariance,
        "innovation_cov",
    ),
]
OUTPUT_IDS = [o[0] for o in OUTPUTS]


class TestOutputs:
    """Each output is labeled with the problem's spaces, of the class and name it says."""

    @pytest.mark.parametrize(
        ("label", "read", "row", "col", "cls", "name"), OUTPUTS, ids=OUTPUT_IDS
    )
    def test_labels_class_and_name(
        self, problem, post, label, read, row, col, cls, name
    ):
        spaces = {"state": problem.prior.space, "obs": problem.obs.space}
        output = read(post)
        assert type(output) is cls, label
        assert output.name == name
        if col is None:
            assert output.space is spaces[row]
        else:
            assert output.rows is spaces[row] and output.cols is spaces[col]

    @pytest.mark.parametrize(
        ("label", "read", "row", "col", "cls", "name"), OUTPUTS, ids=OUTPUT_IDS
    )
    def test_formed_once(self, post, label, read, row, col, cls, name):
        assert read(post) is read(post)

    def test_mean_is_the_estimate(self, post):
        assert post.mean is post.estimate

    def test_estimate_of_a_block(self, problem, post):
        flux = post.estimate["flux"]
        assert flux.space.select("flux").equals(problem.prior.space.select("flux"))

    def test_modeled_obs_include_the_offset(self, problem, post):
        H = problem.forward.dense()
        np.testing.assert_allclose(
            post.modeled_obs.values, H @ post.estimate.values + 399.0
        )
        np.testing.assert_allclose(
            problem.predict(problem.prior).values, H @ problem.prior.values + 399.0
        )

    @pytest.mark.parametrize(
        "attr", ["dofs", "chi2", "reduced_chi2", "r2", "rmse", "uncertainty_reduction"]
    )
    def test_scalars_are_floats(self, post, attr):
        assert isinstance(getattr(post, attr), float)

    def test_reduced_chi2_is_chi2_over_the_observations(self, post):
        assert post.reduced_chi2 == post.chi2 / 8

    def test_scalars_from_their_definitions(self, problem, post):
        z = problem.obs.values
        y = post.modeled_obs.values
        assert post.rmse == pytest.approx(np.sqrt(np.mean((z - y) ** 2)), rel=1e-12)
        assert post.r2 == pytest.approx(np.corrcoef(z, y)[0, 1] ** 2, rel=1e-12)
        assert post.uncertainty_reduction == pytest.approx(
            1
            - np.sqrt(
                np.trace(post.cov().dense()) / np.trace(problem.prior_cov.dense())
            ),
            rel=1e-10,
        )
        assert post.dofs == pytest.approx(
            np.trace(post.averaging_kernel().dense()), rel=1e-10
        )

    def test_dofs_of_an_identity_problem(self):
        """One observation per state element with equal errors: each is half known."""
        space = Space({"x": pd.Index(range(4), name="i")})
        obs = Space({"z": pd.Index(range(4), name="j")})
        post = fips.solve(
            Problem(
                prior=Vector(np.zeros(4), space),
                prior_cov=Covariance.diagonal(1.0, space),
                obs=Vector(np.ones(4), obs),
                obs_cov=Covariance.diagonal(1.0, obs),
                forward=Operator(np.eye(4), obs, space),
            )
        )
        assert post.dofs == pytest.approx(2.0, rel=1e-12)
        np.testing.assert_allclose(post.variances.values, 0.5)
        np.testing.assert_allclose(post.estimate.values, 0.5)

    def test_repr(self, post):
        assert repr(post) == "Posterior(AnalyticOptions(gamma=1.0), n_state=8, n_obs=8)"


class TestDesroziers:
    def test_per_observation(self, problem, post):
        table = post.desroziers()
        assert list(table.columns) == ["diagnosed", "specified", "ratio"]
        assert table.index.equals(problem.obs.index)
        before = problem.obs.values - problem.predict(problem.prior).values
        after = problem.obs.values - post.modeled_obs.values
        np.testing.assert_allclose(table["diagnosed"], after * before)
        np.testing.assert_allclose(table["specified"], np.diag(problem.obs_cov.dense()))
        np.testing.assert_allclose(
            table["ratio"], table["diagnosed"] / table["specified"]
        )

    def test_the_specified_variances_are_divided_by_gamma(self, problem):
        table = fips.solve(problem, gamma=4.0).desroziers()
        np.testing.assert_allclose(
            table["specified"], np.diag(problem.obs_cov.dense()) / 4.0
        )

    def test_by_a_level(self, post):
        per_obs = post.desroziers()
        by_site = post.desroziers(by="site")
        assert by_site.index.tolist() == ["S1", "S2"]
        means = per_obs.groupby("site")[["diagnosed", "specified"]].mean()
        np.testing.assert_allclose(by_site["diagnosed"], means["diagnosed"])
        np.testing.assert_allclose(
            by_site["ratio"], means["diagnosed"] / means["specified"]
        )

    def test_by_a_grouper(self, post):
        monthly = post.desroziers(by=pd.Grouper(level="time", freq="MS"))
        assert len(monthly) == 3
        assert monthly["diagnosed"].notna().all()


class TestLeverage:
    """``leverage(x)`` has the formula ``Estimator.leverage`` had, from workshop slides."""

    @staticmethod
    def _closed_form(H, S_0, S_z, x, c=0.0):
        """The formula, written out with numpy.linalg and no fips code."""
        A = H @ S_0 @ H.T + S_z
        Hx = (H @ x + c)[:, None]
        return (
            Hx @ np.linalg.inv(Hx.T @ np.linalg.solve(A, Hx)) @ Hx.T @ np.linalg.inv(A)
        )

    @staticmethod
    def _arrays(seed=0, m=6, n=4):
        rng = np.random.default_rng(seed)
        H = rng.normal(size=(m, n))
        B, C = rng.normal(size=(n, n)), rng.normal(size=(m, m))
        S_0 = B @ B.T + np.eye(n)
        S_z = C @ C.T + np.eye(m)  # correlated errors
        return H, S_0, S_z, rng.normal(size=m), rng.normal(size=n), rng.normal(size=n)

    @staticmethod
    def _post(H, S_0, S_z, z, x_0, c=None, gamma=1.0):
        state = Space({"flux": pd.Index(range(len(x_0)), name="cell")})
        obs = Space({"co2": pd.Index(range(len(z)), name="obs_id")})
        problem = Problem(
            prior=Vector(x_0, state),
            prior_cov=Covariance(S_0, state),
            obs=Vector(z, obs),
            obs_cov=Covariance(S_z, obs),
            forward=Operator(H, obs, state),
            offset=c,
        )
        return fips.solve(problem, gamma=gamma)

    def test_matches_the_formula(self):
        H, S_0, S_z, z, x_0, x = self._arrays()
        post = self._post(H, S_0, S_z, z, x_0)
        L = post.leverage(Vector(x, post.problem.prior.space))
        assert isinstance(L, Operator) and L.name == "leverage"
        assert L.rows is post.problem.obs.space and L.cols is post.problem.obs.space
        np.testing.assert_allclose(
            L.dense(), self._closed_form(H, S_0, S_z, x), rtol=1e-10
        )

    def test_is_a_projection_with_unit_trace(self):
        H, S_0, S_z, z, x_0, x = self._arrays(seed=3)
        post = self._post(H, S_0, S_z, z, x_0)
        L = post.leverage(Vector(x, post.problem.prior.space)).dense()
        np.testing.assert_allclose(L @ L, L, atol=1e-10)
        assert np.trace(L) == pytest.approx(1.0, abs=1e-10)
        np.testing.assert_allclose(L @ (H @ x), H @ x, atol=1e-10)

    def test_uses_the_offset(self):
        H, S_0, S_z, z, x_0, x = self._arrays(seed=5)
        post = self._post(H, S_0, S_z, z, x_0, c=2.5)
        L = post.leverage(Vector(x, post.problem.prior.space)).dense()
        np.testing.assert_allclose(
            L, self._closed_form(H, S_0, S_z, x, c=2.5), rtol=1e-10
        )

    def test_uses_the_mismatch_divided_by_gamma(self):
        H, S_0, S_z, z, x_0, x = self._arrays(seed=7)
        post = self._post(H, S_0, S_z, z, x_0, gamma=2.0)
        L = post.leverage(Vector(x, post.problem.prior.space)).dense()
        np.testing.assert_allclose(
            L, self._closed_form(H, S_0, S_z / 2.0, x), rtol=1e-10
        )

    def test_the_state_is_matched_by_label(self):
        H, S_0, S_z, z, x_0, x = self._arrays(seed=9)
        post = self._post(H, S_0, S_z, z, x_0)
        space = post.problem.prior.space
        reversed_space = Space({"flux": space.select("flux")[::-1]})
        np.testing.assert_array_equal(
            post.leverage(Vector(x[::-1], reversed_space)).dense(),
            post.leverage(Vector(x, space)).dense(),
        )

    def test_of_the_estimate(self, post):
        L = post.leverage(post.estimate)
        assert np.trace(L.dense()) == pytest.approx(1.0, abs=1e-10)


class TestPickle:
    def test_round_trip(self, post, tmp_path):
        path = tmp_path / "post.pkl"
        post.to_file(path)
        loaded = Posterior.from_file(path)
        assert loaded.options == post.options
        assert loaded.problem.prior.space == post.problem.prior.space
        for _, read, _, col, _, _ in OUTPUTS:
            got, want = read(loaded), read(post)
            if col is None:
                np.testing.assert_array_equal(got.values, want.values)
            else:
                np.testing.assert_array_equal(got.dense(), want.dense())
        assert loaded.dofs == post.dofs and loaded.chi2 == post.chi2

    def test_outputs_are_not_pickled(self, post):
        _ = post.cov(), post.variances, post.kalman_gain
        state = post.__getstate__()
        assert set(state) == {"problem", "options", "factors"}
        loaded = pickle.loads(pickle.dumps(post))
        assert not {"_cov", "variances", "kalman_gain", "estimate"} & set(vars(loaded))


def test_exported():
    assert fips.Problem is Problem
    assert fips.Posterior is Posterior
    assert callable(fips.solve)
    for gone in ("InverseProblem", "Estimator", "available_estimators", "convolve"):
        assert not hasattr(fips, gone)
