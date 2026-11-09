"""The table of the ``Posterior`` docstring: which member each kind of answer gives, held to the code, and the same table on the reference page."""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fips

ROOT = Path(__file__).resolve().parents[1]

MEMBERS = (
    "mean",
    "with_gamma",
    "log_evidence",
    "recovery",
    "data_worth",
    "candidate_worth",
    "at_bound",
    "iterations",
    "eigenvalues",
)

ANALYTIC = dict(zip(MEMBERS, "YYYYYYNNN", strict=True))
"""What a Gaussian posterior of the analytic method gives: Y (yes) or N (raises)."""

# case -> the members it gives, in the order of MEMBERS
EXPECTED = {
    "analytic": ANALYTIC,
    "bounded, no finite bound": ANALYTIC,
    "gauss_newton, an operator": ANALYTIC,
    "bounded": dict(zip(MEMBERS, "YYNNNNYNN", strict=True)),
    "gauss_newton, a model": dict(zip(MEMBERS, "YNYNNNNYN", strict=True)),
    "ensemble": dict(zip(MEMBERS, "YYNNNNNNN", strict=True)),
    "variational, linear, full rank": dict(zip(MEMBERS, "YYYNNNNYY", strict=True)),
    "variational, linear, rank 1": dict(zip(MEMBERS, "YYNNNNNYY", strict=True)),
    "variational, model, full rank": dict(zip(MEMBERS, "NNYNNNNYY", strict=True)),
    "variational, model, rank 1": dict(zip(MEMBERS, "NNNNNNNYY", strict=True)),
}


@pytest.fixture(scope="module")
def posts() -> dict[str, fips.Posterior]:
    """Return one small posterior per case."""
    rng = np.random.default_rng(1)
    state = fips.Space({"flux": pd.Index(list("abcd"), name="cell")})
    observed = fips.Space({"co2": pd.RangeIndex(6, name="obs_id")})
    H = rng.uniform(0, 1, (6, 4))
    forward = fips.Operator(H, observed, state)
    obs = fips.Vector(H @ [1.2, 0.8, 1.5, 0.5] + rng.normal(0, 0.05, 6), observed)

    def problem(forward, prior):
        return fips.Problem(
            prior=fips.Vector(prior, state),
            prior_cov=fips.Independent().with_std(0.5),
            obs=obs,
            obs_cov=fips.Independent().with_std(0.05),
            forward=forward,
        )

    linear = problem(forward, np.ones(4))
    model = problem(fips.LognormalForward(forward, 1.0), np.zeros(4))
    return {
        "analytic": fips.solve(linear),
        "bounded, no finite bound": fips.solve(linear, "bounded", draws=50),
        "gauss_newton, an operator": fips.solve(linear, "gauss_newton", draws=50),
        "bounded": fips.solve(linear, "bounded", lower=1.0, draws=50),
        "gauss_newton, a model": fips.solve(model, "gauss_newton", draws=50),
        "ensemble": fips.solve(linear, "ensemble", members=50),
        "variational, linear, full rank": fips.solve(linear, "variational", rank=4),
        "variational, linear, rank 1": fips.solve(linear, "variational", rank=1),
        "variational, model, full rank": fips.solve(model, "variational", rank=4),
        "variational, model, rank 1": fips.solve(model, "variational", rank=1),
    }


def ask(post: fips.Posterior, member: str, linear: fips.Problem):
    """Ask `post` for `member`."""
    total = fips.Operator(
        np.ones((1, 4)),
        fips.Space({"all": pd.Index(["all"], name="region")}),
        post.problem.prior.space,
    )
    if member == "with_gamma":
        return post.with_gamma(2.0)
    if member == "recovery":
        return post.recovery(linear)
    if member == "data_worth":
        return post.data_worth("obs_id", total)
    if member == "candidate_worth":
        candidates = fips.Operator(
            np.full((2, 4), 0.5),
            fips.Space({"co2": pd.RangeIndex(10, 12, name="obs_id")}),
            post.problem.prior.space,
        )
        return post.candidate_worth(
            candidates, fips.Independent().with_std(0.05), "obs_id", total
        )
    return getattr(post, member)


@pytest.mark.parametrize("case", EXPECTED)
def test_the_table_is_what_each_posterior_gives(case, posts):
    """A member the table marks as raising raises UnavailableError; the others work."""
    linear = posts["analytic"].problem
    for member, answer in EXPECTED[case].items():
        if answer == "Y":
            ask(posts[case], member, linear)
        else:
            with pytest.raises(fips.UnavailableError):
                ask(posts[case], member, linear)


def table(text: str) -> list[str]:
    """Return the lines of the first ``list-table`` in `text`, dedented."""
    lines = inspect.cleandoc(text).splitlines() if "\n" in text else [text]
    start = next(i for i, line in enumerate(lines) if line.strip() == ".. list-table::")
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for line in lines[start:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent and block:
            break
        block.append(line[indent:] if line.strip() else "")
    while block and not block[-1]:
        block.pop()
    return block


def test_the_reference_page_has_the_same_table():
    """The table is written twice (the docstring and ``docs/reference/posterior.rst``); one must not drift from the other."""
    page = (ROOT / "docs" / "reference" / "posterior.rst").read_text()
    assert table(page) == table(fips.Posterior.__doc__ or "")


TOTAL_ROWS = fips.Space({"all": pd.Index(["all"], name="region")})


@pytest.mark.parametrize(
    "case",
    [
        "analytic",
        "ensemble",
        "variational, linear, rank 1",
        "variational, model, rank 1",
    ],
)
def test_an_interval_that_is_not_sampled_is_the_estimate_plus_or_minus_z_sd(
    case, posts
):
    """A posterior with no kept draws reads its interval as the Gaussian's, about its estimate."""
    post = posts[case]
    total = fips.Operator(np.ones((1, 4)), TOTAL_ROWS, post.problem.prior.space)
    for W, center in (
        (None, post.estimate.values),
        (total, [post.estimate.values.sum()]),
    ):
        table = post.uncertainty(W, level=0.9)
        z = 1.6448536269514722  # the normal's 0.95 quantile
        np.testing.assert_allclose(
            table["lower"], center - z * table["posterior_std"], rtol=1e-12
        )
        np.testing.assert_allclose(
            table["upper"], center + z * table["posterior_std"], rtol=1e-12
        )


@pytest.mark.parametrize("case", ["bounded", "gauss_newton, a model"])
def test_a_sampled_answer_reads_its_mean_and_interval_from_its_draws(case, posts):
    """A sampled posterior's mean is the kept draws' mean (not the mode), and its interval their quantiles, of the state or of a total."""
    post = posts[case]
    kept = post._solution.kept()
    np.testing.assert_array_equal(post.mean.values, kept.mean(axis=1))
    assert post.mean.space == post.estimate.space
    assert post.mean is post.mean  # kept
    total = fips.Operator(np.ones((1, 4)), TOTAL_ROWS, post.problem.prior.space)
    for W, draws in ((None, kept), (total, kept.sum(axis=0, keepdims=True))):
        table = post.uncertainty(W, level=0.8)
        np.testing.assert_allclose(
            table["lower"], np.quantile(draws, 0.1, axis=1), rtol=1e-12
        )
        np.testing.assert_allclose(
            table["upper"], np.quantile(draws, 0.9, axis=1), rtol=1e-12
        )
        # the draws' standard deviation, as before
        np.testing.assert_allclose(
            table["posterior_std"], draws.std(axis=1, ddof=1), rtol=1e-10
        )


@pytest.mark.parametrize("level", [0.0, 1.0, -0.5, 1.5, True, "0.95", None])
def test_an_interval_needs_a_probability(level, posts):
    """``level`` is a number strictly between 0 and 1, in uncertainty and in trials."""
    with pytest.raises(ValueError, match="level is a probability"):
        posts["analytic"].uncertainty(level=level)
    with pytest.raises(ValueError, match="level is a probability"):
        posts["analytic"].trials(n=1, rng=0, level=level)
