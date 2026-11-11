"""
fips's warnings are Python warnings too: a :class:`fips.FipsWarning`, pointed at the user's line, besides the log line.

The ``fips`` logger has a NullHandler, so a log line alone reaches no screen
unless the program configures logging. Here: a warning is issued with
logging left as it is, at the line of this file that called fips, with the
same text as the log line; a filter makes it an error; and a solve with
nothing to warn about issues none (INFO lines stay log-only).
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

import fips


def low_rank_problem() -> fips.Problem:
    """A 6 x 10 problem whose data inform 6 directions: ``rank=2`` leaves out four that still hold information."""
    state = fips.Space({"x": pd.RangeIndex(10, name="i")})
    observed = fips.Space({"z": pd.RangeIndex(6, name="j")})
    H = np.random.default_rng(0).normal(size=(6, 10))
    return fips.Problem(
        prior=fips.Vector(np.zeros(10), state),
        prior_cov=fips.Independent().with_std(1.0),
        obs=fips.Vector(np.ones(6), observed),
        obs_cov=fips.Independent().with_std(0.1),
        forward=fips.Operator(H, observed, state),
    )


def test_fips_warning_is_a_public_user_warning():
    """``fips.FipsWarning`` is exported and a :class:`UserWarning`, so Python shows it by default."""
    assert "FipsWarning" in fips.__all__
    assert issubclass(fips.FipsWarning, UserWarning)


def test_a_warning_reaches_python_without_logging_configured():
    """The variational rank warning is a FipsWarning with the fips logger left with its NullHandler alone."""
    handlers = logging.getLogger("fips").handlers
    assert all(isinstance(h, logging.NullHandler) for h in handlers)
    with pytest.warns(fips.FipsWarning, match="Raise rank"):
        fips.solve(low_rank_problem(), "variational", rank=2)


def test_a_warning_points_at_the_users_line():
    """The warning's file is the caller's (this test), not the fips module that found it."""
    problem = low_rank_problem()
    with pytest.warns(fips.FipsWarning) as caught:
        fips.solve(problem, "variational", rank=2)
    assert [w.filename for w in caught] == [__file__]


def test_a_warning_is_still_logged(caplog):
    """The log line stays, with the warning's text, under the method's logger."""
    with (
        caplog.at_level(logging.WARNING, logger="fips"),
        pytest.warns(fips.FipsWarning) as caught,
    ):
        fips.solve(low_rank_problem(), "variational", rank=2)
    records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert [r.name for r in records] == ["fips.methods.variational"]
    assert [r.getMessage() for r in records] == [str(w.message) for w in caught]


def test_a_filter_makes_a_warning_an_error():
    """``warnings.filterwarnings("error", category=fips.FipsWarning)`` turns the warning into an exception."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", fips.FipsWarning)
        with pytest.raises(fips.FipsWarning, match="Raise rank"):
            fips.solve(low_rank_problem(), "variational", rank=2)


def test_a_solve_with_nothing_to_warn_about_issues_no_warning(caplog):
    """At full rank the variational solve logs INFO lines and issues no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level(logging.INFO, logger="fips"):
            fips.solve(low_rank_problem(), "variational", rank=6)
    assert any(r.levelno == logging.INFO for r in caplog.records)
    assert caught == []
