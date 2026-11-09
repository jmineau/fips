"""
The ensemble method: its square-root update, its convergence to the analytic posterior, and its interface.

An ensemble cannot meet the oracle's 1e-10 (``tests/test_oracle.py`` leaves
out a method registered with ``exact=False``). It is held to being right
three ways instead: its update is exact for its own ensemble covariance (to
roundoff, against the Kalman update written here with numpy); its mean and
covariance approach the analytic posterior as the members grow, the error
falling as ``1/sqrt(N)``; and its twins (``with_data``, ``with_gamma``, the
two-stage and the shortcut paths, labels in another order, localization
weights of one) give the same posterior.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.linalg import sqrtm

import fips
from fips.methods.ensemble import EnsembleOptions

N_STATE, N_OBS = 8, 12


def build(
    seed: int = 0,
    n: int = N_STATE,
    m: int = N_OBS,
    dense_mismatch: bool = False,
    forward=None,
) -> fips.Problem:
    """A small explicit-H problem: correlated prior, observations of a drawn truth."""
    rng = np.random.default_rng(seed)
    state = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(m, name="obs_id")})
    H = fips.Operator(rng.normal(size=(m, n)), observed, state)
    if dense_mismatch:
        obs_cov = fips.Exponential("obs_id", 1.5).with_std(0.5) + (
            fips.Independent().with_std(0.2)
        )
    else:
        obs_cov = fips.Independent().with_std(
            pd.Series(rng.uniform(0.3, 0.8, m), index=observed.select("co2"))
        )
    problem = fips.Problem(
        prior=fips.Vector(rng.normal(size=n), state),
        prior_cov=fips.Exponential("cell", 2.0).with_std(1.0),
        obs=fips.Vector(np.zeros(m), observed),
        obs_cov=obs_cov,
        forward=H if forward is None else forward(H),
        offset=0.3,
    )
    _, obs = problem.simulate(rng=rng)
    return fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward,
        offset=problem.offset,
    )


class Cubic(fips.ForwardModel):
    """A model with no Jacobian: ``H x + 0.1 (H x)^3``, counting its runs."""

    def __init__(self, H):
        super().__init__(H.rows, H.cols)
        self.H = H
        self.runs = 0

    def __call__(self, x):
        self.runs += 1
        y = self.H @ x
        return y + 0.1 * (y * y * y)


def modeled(problem: fips.Problem, members: pd.DataFrame) -> pd.DataFrame:
    """Run the problem's forward on each member, as a user would outside fips."""
    columns = {
        label: problem.predict(fips.Vector.from_series(members[label])).values
        - problem._offset_values()
        for label in members.columns
    }
    return pd.DataFrame(columns, index=problem.obs.index)


def deviations(problem: fips.Problem, members: pd.DataFrame, predicted: pd.DataFrame):
    """Return the scaled deviations A and B, in the problem's orders."""
    X = members.loc[problem.prior.index].to_numpy()
    Y = predicted.loc[problem.obs.index, members.columns].to_numpy()
    scale = np.sqrt(X.shape[1] - 1)
    return (
        (X - X.mean(axis=1, keepdims=True)) / scale,
        (Y - Y.mean(axis=1, keepdims=True)) / scale,
    )


def kalman_update(A, B, R):
    """The Kalman posterior of the ensemble's own covariance, ``P - P_xy (P_yy + R)^-1 P_yx``."""
    P, P_xy, P_yy = A @ A.T, A @ B.T, B @ B.T
    return P - P_xy @ np.linalg.solve(P_yy + R, P_xy.T)


def updated_cov(post: fips.Posterior) -> np.ndarray:
    """The covariance of the updated members, by label (``post.cov()``)."""
    return post.cov().dense()


# ---------------------------------------------------------------------------
# The square-root update is exact for the ensemble's own covariance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dense_mismatch", [False, True], ids=["diagonal", "dense"])
@pytest.mark.parametrize("model", ["linear", "cubic"])
@pytest.mark.parametrize("N", [5, 40])
@pytest.mark.parametrize("gamma", [1.0, 0.4])
def test_square_root_is_the_kalman_update_of_its_own_covariance(
    dense_mismatch, model, N, gamma
):
    """The updated deviations' covariance is the Kalman posterior of the forecast ensemble's, to roundoff, for N below and above m."""
    problem = build(
        1, dense_mismatch=dense_mismatch, forward=Cubic if model == "cubic" else None
    )
    members = problem.ensemble(N, rng=2)
    predicted = modeled(problem, members)
    post = fips.solve(
        problem, "ensemble", members=members, predicted=predicted, gamma=gamma
    )
    A, B = deviations(problem, members, predicted)
    R = problem.obs_cov.dense() / gamma
    expected = kalman_update(A, B, R)
    scale = np.abs(A @ A.T).max()
    np.testing.assert_allclose(updated_cov(post), expected, rtol=0, atol=1e-11 * scale)
    # the mean moves by the ensemble gain
    d = (
        problem.obs.values
        - problem._offset_values()
        - predicted.to_numpy().mean(axis=1)
    )
    K = (A @ B.T) @ np.linalg.inv(B @ B.T + R)
    np.testing.assert_allclose(
        post.estimate.values, members.to_numpy().mean(axis=1) + K @ d, atol=1e-11
    )
    np.testing.assert_allclose(
        post.kalman_gain.dense(), K, atol=1e-11 * np.abs(K).max()
    )
    # and the members' mean is the estimate
    np.testing.assert_allclose(
        post.sample(N, rng=0).mean(axis=1).to_numpy(), post.estimate.values, atol=1e-11
    )


def test_wrong_updates_fail_the_exactness_check():
    """The check above catches a perturbed-observation update without perturbations, and deviations left unchanged."""
    problem = build(1)
    members = problem.ensemble(40, rng=2)
    predicted = modeled(problem, members)
    A, B = deviations(problem, members, predicted)
    R = problem.obs_cov.dense()
    expected = kalman_update(A, B, R)
    scale = np.abs(A @ A.T).max()
    K = (A @ B.T) @ np.linalg.inv(B @ B.T + R)
    no_perturbations = A - K @ B  # (I - KH) P (I - KH)^T: too small
    for wrong in (no_perturbations, A):
        error = np.abs(wrong @ wrong.T - expected).max()
        assert error > 1e-3 * scale
    right = fips.solve(problem, "ensemble", members=members, predicted=predicted)
    assert np.abs(updated_cov(right) - expected).max() < 1e-11 * scale


def test_whitened_reduced_gain_matches_whitaker_and_hamill_with_scalar_mismatch():
    """With localization and a mismatch that is a multiple of I, the update is Whitaker and Hamill's batch reduced gain with symmetric roots."""
    rng = np.random.default_rng(5)
    n, m, N = 10, 9, 30
    state = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(m, name="obs_id")})
    H = fips.Operator(rng.normal(size=(m, n)), observed, state)
    problem = fips.Problem(
        prior=fips.Vector(np.zeros(n), state),
        prior_cov=fips.Exponential("cell", 3.0).with_std(1.0),
        obs=fips.Vector(rng.normal(size=m), observed),
        obs_cov=fips.Independent().with_std(0.4),
        forward=H,
    )
    cells, sites = np.arange(n) * 1.0, np.linspace(0, n - 1, m)
    rho_xy = np.exp(-(((cells[:, None] - sites[None, :]) / 3.0) ** 2))
    rho_yy = np.exp(-(((sites[:, None] - sites[None, :]) / 3.0) ** 2))
    members = problem.ensemble(N, rng=1)
    predicted = modeled(problem, members)
    post = fips.solve(
        problem,
        "ensemble",
        members=members,
        predicted=predicted,
        localization=fips.Operator(rho_xy, state, observed),
        obs_localization=fips.Operator(rho_yy, observed, observed),
    )
    A, B = deviations(problem, members, predicted)
    R = 0.16 * np.eye(m)
    P_xy, G = rho_xy * (A @ B.T), rho_yy * (B @ B.T) + R
    root_G, root_R = np.real(sqrtm(G)), np.real(sqrtm(R))
    reduced = P_xy @ np.linalg.inv(root_G) @ np.linalg.inv(root_G + root_R)
    expected = (A - reduced @ B) @ (A - reduced @ B).T
    np.testing.assert_allclose(updated_cov(post), expected, atol=1e-11)
    d = problem.obs.values - predicted.to_numpy().mean(axis=1)
    np.testing.assert_allclose(
        post.estimate.values, P_xy @ np.linalg.solve(G, d), atol=1e-11
    )


# ---------------------------------------------------------------------------
# Convergence to the analytic posterior
# ---------------------------------------------------------------------------

SIZES = [25, 50, 100, 200, 400, 800, 1600, 3200]


def errors(problem: fips.Problem, N: int, seeds) -> tuple[float, float]:
    """The root mean square, over seeds, of the ensemble's mean error (in posterior standard deviations) and its covariance error (relative, Frobenius)."""
    exact = fips.solve(problem)
    std = np.sqrt(exact.variances.values)
    S_hat = exact.cov().dense()
    mean_errors, cov_errors = [], []
    for seed in seeds:
        post = fips.solve(problem, "ensemble", members=N, seed=seed)
        mean_errors.append(
            np.mean(((post.estimate.values - exact.estimate.values) / std) ** 2)
        )
        cov_errors.append(
            np.linalg.norm(updated_cov(post) - S_hat) ** 2 / np.linalg.norm(S_hat) ** 2
        )
    return float(np.sqrt(np.mean(mean_errors))), float(np.sqrt(np.mean(cov_errors)))


def test_mean_and_covariance_approach_the_analytic_ones_as_one_over_root_n():
    """
    Over N = 25 to 3200, the errors of the mean and of the covariance fall with a fitted log-log slope of about -1/2.

    Over the whole range the slope is a little steeper (about -0.55 to -0.6
    over eight problems): at 25 members the gain's error has a part that
    falls as 1/N. From 200 members on it is -1/2 within 0.1.
    """
    problem = build(3)
    table = np.array([errors(problem, N, range(32)) for N in SIZES])
    log_n = np.log(SIZES)
    for column, what in enumerate(("mean", "covariance")):
        log_error = np.log(table[:, column])
        whole = np.polyfit(log_n, log_error, 1)[0]
        late = np.polyfit(log_n[3:], log_error[3:], 1)[0]
        assert -0.65 < whole < -0.45, (what, whole, table[:, column])
        assert -0.6 < late < -0.4, (what, late, table[:, column])
    # at 3200 members the mean is within a few hundredths of a standard deviation
    assert table[-1, 0] < 0.05


# ---------------------------------------------------------------------------
# Twins
# ---------------------------------------------------------------------------


def assert_same(post: fips.Posterior, other: fips.Posterior, atol=1e-11):
    """The two posteriors have the same estimate, covariance, gain and diagnostics, by label."""
    np.testing.assert_allclose(post.estimate.values, other.estimate.values, atol=atol)
    np.testing.assert_allclose(updated_cov(post), updated_cov(other), atol=atol)
    np.testing.assert_allclose(
        post.kalman_gain.dense(), other.kalman_gain.dense(), atol=atol
    )
    for name in ("dofs", "information_content", "chi2", "data_misfit", "state_misfit"):
        assert getattr(post, name) == pytest.approx(
            getattr(other, name), rel=1e-10, abs=atol
        )


@pytest.mark.parametrize("model", ["linear", "cubic"])
def test_two_stage_and_shortcut_agree(model):
    """problem.ensemble, the model run by hand and predicted= give what members=N gives."""
    problem = build(4, forward=Cubic if model == "cubic" else None)
    members = problem.ensemble(30, rng=7)
    two_stage = fips.solve(
        problem, "ensemble", members=members, predicted=modeled(problem, members)
    )
    shortcut = fips.solve(problem, "ensemble", members=30, seed=7)
    given_members = fips.solve(problem, "ensemble", members=members)
    assert_same(two_stage, shortcut)
    assert_same(two_stage, given_members)


def test_labels_in_another_order():
    """The members' rows and the modeled observations' rows and columns are matched by label."""
    problem = build(6)
    members = problem.ensemble(20, rng=3)
    predicted = modeled(problem, members)
    post = fips.solve(problem, "ensemble", members=members, predicted=predicted)
    rng = np.random.default_rng(0)
    shuffled_members = members.iloc[rng.permutation(N_STATE)]
    shuffled_predicted = predicted.iloc[
        rng.permutation(N_OBS), rng.permutation(predicted.shape[1])
    ]
    other = fips.solve(
        problem, "ensemble", members=shuffled_members, predicted=shuffled_predicted
    )
    assert_same(post, other)


def test_with_data_and_with_gamma_are_fresh_solves():
    """with_data and with_gamma reuse the factors and give what a new solve with the same members gives."""
    problem = build(8)
    members = problem.ensemble(25, rng=1)
    predicted = modeled(problem, members)
    post = fips.solve(problem, "ensemble", members=members, predicted=predicted)
    _, obs = problem.simulate(rng=11)
    new = fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward,
        offset=problem.offset,
    )
    fresh = fips.solve(new, "ensemble", members=members, predicted=predicted)
    assert_same(post.with_data(obs), fresh)
    for gamma in (0.3, 4.0):
        fresh = fips.solve(
            problem, "ensemble", members=members, predicted=predicted, gamma=gamma
        )
        assert_same(post.with_gamma(gamma), fresh)


def test_localization_of_ones_is_none():
    """Weights of one everywhere give the unlocalized update, its members too."""
    problem = build(9)
    members = problem.ensemble(15, rng=4)
    predicted = modeled(problem, members)
    state, observed = problem.prior.space, problem.obs.space
    plain = fips.solve(problem, "ensemble", members=members, predicted=predicted)
    ones = fips.solve(
        problem,
        "ensemble",
        members=members,
        predicted=predicted,
        localization=fips.Operator(np.ones((N_STATE, N_OBS)), state, observed),
        obs_localization=fips.Operator(np.ones((N_OBS, N_OBS)), observed, observed),
    )
    assert_same(plain, ones)
    np.testing.assert_allclose(
        plain.sample(15, rng=0).to_numpy(),
        ones.sample(15, rng=0).to_numpy(),
        atol=1e-11,
    )
    assert plain.chi2_tests().equals(ones.chi2_tests()) or np.allclose(
        plain.chi2_tests()[["chi2", "dof"]], ones.chi2_tests()[["chi2", "dof"]]
    )


def test_a_cell_no_observation_is_near_keeps_its_prior_members():
    """A state element whose weights are all zero is not updated: its members, mean and spread are the prior's."""
    problem = build(10)
    members = problem.ensemble(20, rng=5)
    state, observed = problem.prior.space, problem.obs.space
    rho_xy = np.ones((N_STATE, N_OBS))
    rho_xy[2] = 0.0
    post = fips.solve(
        problem,
        "ensemble",
        members=members,
        localization=fips.Operator(rho_xy, state, observed),
        obs_localization=fips.Operator(np.eye(N_OBS), observed, observed),
    )
    np.testing.assert_allclose(
        post.sample(20, rng=0).iloc[2], members.iloc[2], atol=1e-12
    )
    assert post.estimate.values[2] == pytest.approx(problem.prior.values[2], abs=1e-12)


# ---------------------------------------------------------------------------
# What the posterior gives
# ---------------------------------------------------------------------------


def test_diagnostics_are_the_analytic_formulas_with_the_ensemble_covariance():
    """DOFS, the information content and chi-squared use P_yy for H S_0 H^T; with a linear model DOFS is tr(K H)."""
    problem = build(12)
    members = problem.ensemble(6, rng=0)
    predicted = modeled(problem, members)
    post = fips.solve(
        problem, "ensemble", members=members, predicted=predicted, gamma=2.0
    )
    A, B = deviations(problem, members, predicted)
    R = problem.obs_cov.dense() / 2.0
    G = B @ B.T + R
    d = (
        problem.obs.values
        - problem._offset_values()
        - predicted.to_numpy().mean(axis=1)
    )
    assert post.chi2 == pytest.approx(d @ np.linalg.solve(G, d), rel=1e-10)
    assert post.dofs == pytest.approx(np.trace(B @ B.T @ np.linalg.inv(G)), rel=1e-10)
    assert post.dofs == pytest.approx(post.dofs_per_element.values.sum(), rel=1e-10)
    information = 0.5 * (np.linalg.slogdet(G)[1] - np.linalg.slogdet(R)[1])
    assert post.information_content == pytest.approx(information, rel=1e-10)
    assert post.chi2 == pytest.approx(
        2.0 * post.data_misfit + post.state_misfit, rel=1e-10
    )
    tests = post.chi2_tests()
    assert tests.loc["increment", "dof"] == 5  # N - 1 directions, fewer than n and m
    assert post.mean is post.estimate


def test_sample_returns_members_and_stops_at_the_ensemble_size():
    """sample(n) is n of the updated members; all of them have the posterior's mean and covariance; more raise."""
    problem = build(13)
    post = fips.solve(problem, "ensemble", members=12, seed=0)
    draws = post.sample(12, rng=1)
    np.testing.assert_allclose(draws.mean(axis=1), post.estimate.values, atol=1e-12)
    np.testing.assert_allclose(np.cov(draws.to_numpy()), updated_cov(post), atol=1e-12)
    assert post.sample(5, rng=2).shape == (N_STATE, 5)
    with pytest.raises(ValueError, match="12 members"):
        post.sample(13)


def test_trials_reuse_the_modeled_ensemble():
    """Trials run the model no more than the solve did: each trial is with_data on the same members."""
    model_holder = {}

    def counted(H):
        model_holder["model"] = Cubic(H)
        return model_holder["model"]

    problem = build(14, forward=counted)
    before = model_holder["model"].runs  # build ran it to make the data
    post = fips.solve(problem, "ensemble", members=20, seed=0)
    runs = model_holder["model"].runs
    assert runs == before + 20
    world = fips.Problem(
        prior=problem.prior,
        prior_cov=problem.prior_cov,
        obs=problem.obs,
        obs_cov=problem.obs_cov,
        forward=problem.forward.H,
        offset=problem.offset,
    )
    table = post.trials(world, n=30, rng=0)
    assert model_holder["model"].runs == runs
    assert len(table) == 30 * N_STATE
    # every trial claims the same spread: the members do not change
    spread = table["posterior_std"].unstack("trial")
    np.testing.assert_allclose(
        spread.to_numpy(),
        np.repeat(spread.iloc[:, :1].to_numpy(), 30, axis=1),
        rtol=1e-12,
    )


def test_refused_features():
    """The ensemble refuses recovery and log_evidence; the averaging kernel needs H."""
    post = fips.solve(build(15), "ensemble", members=10)
    with pytest.raises(TypeError, match="estimate_is_linear"):
        post.recovery()
    with pytest.raises(TypeError):
        _ = post.log_evidence
    model_post = fips.solve(build(15, forward=Cubic), "ensemble", members=10)
    with pytest.raises(TypeError, match="linear forward operator"):
        _ = model_post.dofs_per_element
    assert np.isfinite(model_post.dofs)


def test_pickles():
    """A posterior read back gives the same numbers, its updated deviations computed again."""
    post = fips.solve(build(16), "ensemble", members=10)
    _ = post.variances
    again = pickle.loads(pickle.dumps(post))
    assert_same(post, again)


def test_model_without_jacobian():
    """A model with only __call__ solves with the ensemble; gauss_newton and taylortest say it has no Jacobian."""
    problem = build(17, forward=Cubic)
    fips.solve(problem, "ensemble", members=10)
    with pytest.raises(NotImplementedError, match="has no jacobian"):
        fips.solve(problem, "gauss_newton")
    with pytest.raises(NotImplementedError, match="has no jacobian"):
        fips.taylortest(problem.forward, problem.prior)


# ---------------------------------------------------------------------------
# The prior ensemble and the options
# ---------------------------------------------------------------------------


def test_problem_ensemble_is_centered_on_the_prior():
    """The members' mean is the prior exactly, their covariance the prior's on average, their labels the prior's."""
    problem = build(18)
    members = problem.ensemble(20000, rng=0)
    assert members.index.equals(problem.prior.index)
    assert members.columns.name == "member"
    np.testing.assert_allclose(members.mean(axis=1), problem.prior.values, atol=1e-12)
    S_0 = problem.prior_cov.dense()
    np.testing.assert_allclose(np.cov(members.to_numpy()), S_0, atol=0.05)
    pd.testing.assert_frame_equal(
        members.iloc[:, :5], problem.ensemble(20000, rng=0).iloc[:, :5]
    )
    with pytest.raises(ValueError, match="at least 2"):
        problem.ensemble(1)


@pytest.mark.parametrize(
    "options, error, match",
    [
        ({}, ValueError, "needs members="),
        ({"members": 1}, ValueError, "at least 2"),
        ({"members": 2.5}, TypeError, "number of members"),
        (
            {"members": 10, "predicted": pd.DataFrame()},
            ValueError,
            "needs members= as the DataFrame",
        ),
        (
            {"members": 10, "predicted": np.zeros((2, 2))},
            TypeError,
            "predicted is a DataFrame",
        ),
        ({"members": 10, "seed": -1}, ValueError, "seed"),
        ({"members": 10, "gamma": 0.0}, ValueError, "gamma"),
    ],
)
def test_options_are_checked(options, error, match):
    """EnsembleOptions refuses what it cannot use, saying why."""
    with pytest.raises(error, match=match):
        EnsembleOptions(**options)


def test_labels_and_localization_are_checked():
    """Modeled observations on other labels, one localization without the other, and weights that are not positive semidefinite raise."""
    problem = build(19)
    members = problem.ensemble(10, rng=0)
    predicted = modeled(problem, members)
    with pytest.raises(ValueError, match="columns of predicted"):
        fips.solve(
            problem, "ensemble", members=members, predicted=predicted.iloc[:, :9]
        )
    with pytest.raises(ValueError, match="rows of predicted"):
        fips.solve(problem, "ensemble", members=members, predicted=predicted.iloc[:5])
    with pytest.raises(ValueError, match="columns"):
        fips.solve(problem, "ensemble", members=members.iloc[:, :1])
    state, observed = problem.prior.space, problem.obs.space
    rho_xy = fips.Operator(np.ones((N_STATE, N_OBS)), state, observed)
    with pytest.raises(ValueError, match="both weights"):
        fips.solve(problem, "ensemble", members=members, localization=rho_xy)
    indefinite = np.ones((N_OBS, N_OBS)) - 2.0 * np.eye(N_OBS)
    with pytest.raises(ValueError, match="negative eigenvalue"):
        fips.solve(
            problem,
            "ensemble",
            members=members,
            localization=rho_xy,
            obs_localization=fips.Operator(indefinite, observed, observed),
        )
    assert "<8 x 10 DataFrame>" in repr(
        fips.solve(problem, "ensemble", members=members).options
    )


def gaspari_cohn(distance: np.ndarray, half_width: float) -> np.ndarray:
    """Gaspari and Cohn's (1999) fifth-order function, zero beyond twice the half-width."""
    r = np.abs(distance) / half_width
    w = np.zeros_like(r)
    a, b = r <= 1.0, (r > 1.0) & (r < 2.0)
    ra, rb = r[a], r[b]
    w[a] = -(ra**5) / 4 + ra**4 / 2 + 5 * ra**3 / 8 - 5 * ra**2 / 3 + 1
    w[b] = (
        rb**5 / 12
        - rb**4 / 2
        + 5 * rb**3 / 8
        + 5 * rb**2 / 3
        - 5 * rb
        + 4
        - 2 / (3 * rb)
    )
    return w


def test_localization_restores_the_coverage_of_a_small_ensemble():
    """
    In a world with short correlations, 20 members' intervals of regional totals cover far too rarely, and localized ones about as often as they claim.

    The user guide's world, smaller: 60 cells on a line, 20 observations of
    the cells upwind, a prior correlated over 2 cells; 300 trials of the
    problem's own world, each solved with the same members.
    """
    n, m = 60, 20
    cell_x, site_x = np.arange(n, dtype=float), np.linspace(2.0, n - 3.0, m)
    upwind = site_x[:, None] - cell_x[None, :]
    footprint = np.where(upwind >= 0, np.exp(-upwind / 3.0), 0.0)
    cells = fips.Space({"flux": pd.RangeIndex(n, name="cell")})
    observed = fips.Space({"ch4": pd.RangeIndex(m, name="obs_id")})
    problem = fips.Problem(
        prior=fips.Vector(np.ones(n), cells),
        prior_cov=fips.Exponential("cell", 2.0).with_std(0.5),
        obs=fips.Vector(np.zeros(m), observed),
        obs_cov=fips.Independent().with_std(0.2),
        forward=fips.Operator(footprint, observed, cells),
    )
    regions = fips.Operator(
        np.kron(np.eye(3), np.ones((1, n // 3))),
        fips.Space({"total": pd.RangeIndex(3, name="region")}),
        cells,
    )
    weights = {
        "localization": fips.Operator(
            gaspari_cohn(cell_x[:, None] - site_x[None, :], 10.0), cells, observed
        ),
        "obs_localization": fips.Operator(
            gaspari_cohn(site_x[:, None] - site_x[None, :], 10.0), observed, observed
        ),
    }

    def coverage(post):
        table = post.trials(n=300, rng=5, W=regions)
        error = (table["estimate"] - table["truth"]).abs()
        return float((error <= 1.96 * table["posterior_std"]).mean())

    exact = coverage(fips.solve(problem))
    plain = coverage(fips.solve(problem, "ensemble", members=20, seed=3))
    local = coverage(fips.solve(problem, "ensemble", members=20, seed=3, **weights))
    assert abs(exact - 0.95) < 0.03
    assert plain < 0.8
    assert abs(local - 0.95) < 0.04, (exact, plain, local)
