"""Test suite for fips.pipeline module."""

import logging
import re

import numpy as np
import pandas as pd
import pytest

from fips import Covariance, Operator, Problem, Vector
from fips.pipeline import InversionPipeline

# What get_inputs says about each step: one line as the step starts, one as it
# ends with its elapsed time. Hooks that can do nothing say so ("Optionally").
TIME = r"\d+\.\d\ds"
GET_INPUTS_MESSAGES = [
    "Loading observations...",
    f"Observations loaded in {TIME}",
    "Loading prior...",
    f"Prior loaded in {TIME}",
    r"Loading constant term \(if any\)\.\.\.",
    f"Constant term loaded in {TIME}",
    "Optionally, filtering state space...",
    f"Optionally, state space filtered in {TIME}",
    "Building forward operator...",
    f"Forward operator built in {TIME}",
    "Building prior error covariance...",
    f"Prior error covariance built in {TIME}",
    "Building model-data mismatch covariance...",
    f"Model-data mismatch covariance built in {TIME}",
    r"Optionally, aggregating observation space \(if needed\)\.\.\.",
    f"Optionally, observation space aggregated in {TIME}",
]

# run wraps get_inputs in its own lines, then initializes and solves.
RUN_MESSAGES = [
    "Getting problem inputs...",
    *GET_INPUTS_MESSAGES,
    f"Inputs prepared in {TIME}",
    "Initializing solver...",
    f"Solver initialized in {TIME}",
    "Solving...",
    f"Solve completed in {TIME}",
    f"Total pipeline time: {TIME}",
]


class _Pipeline(InversionPipeline):
    """A pipeline for a three-element state observed four times."""

    def __init__(self):
        super().__init__(config=None, problem=Problem)
        self.state_idx = pd.Index(["s0", "s1", "s2"], name="state_id")
        self.obs_idx = pd.Index(["o0", "o1", "o2", "o3"], name="obs_id")

    def get_obs(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0], index=self.obs_idx, name="obs")
        return Vector.from_series(series, block="obs")

    def get_prior(self):
        series = pd.Series([0.0, 0.0, 0.0], index=self.state_idx, name="prior")
        return Vector.from_series(series, block="state")

    def get_forward_operator(self, obs, prior):
        return Operator(np.full((len(obs), len(prior)), 0.5), obs.space, prior.space)

    def get_prior_error(self, prior):
        return Covariance.diagonal(1.0, prior.space)

    def get_modeldata_mismatch(self, obs):
        return Covariance.diagonal(2.0, obs.space)


def _pipeline_messages(caplog):
    """The records the pipeline module logged, as ``(level, message)`` pairs."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == "fips.pipeline"
    ]


def _assert_steps(messages, expected):
    """Each message is an INFO record matching its pattern, in order."""
    assert len(messages) == len(expected)
    for (level, message), pattern in zip(messages, expected, strict=True):
        assert level == logging.INFO, message
        assert re.fullmatch(pattern, message), message


class TestStepLogging:
    """The steps of get_inputs and run are logged, not printed."""

    def test_get_inputs_logs_each_step(self, caplog):
        """Every step logs a start line and a timed end line, at INFO."""
        with caplog.at_level(logging.INFO, logger="fips.pipeline"):
            inputs = _Pipeline().get_inputs()

        _assert_steps(_pipeline_messages(caplog), GET_INPUTS_MESSAGES)
        assert set(inputs) == {
            "obs",
            "prior",
            "forward",
            "prior_cov",
            "obs_cov",
            "offset",
        }

    def test_run_logs_each_step(self, caplog):
        """The steps of get_inputs are logged inside those of run, in order."""
        pipeline = _Pipeline()

        with caplog.at_level(logging.INFO, logger="fips.pipeline"):
            post = pipeline.run()

        _assert_steps(_pipeline_messages(caplog), RUN_MESSAGES)
        assert post is pipeline.posterior
        assert post.problem is pipeline.problem
        assert len(pipeline.problem.obs) == 4

    @pytest.mark.parametrize("method", ["get_inputs", "run"])
    def test_nothing_is_printed(self, method, capsys, caplog):
        """Neither method writes anything to stdout or stderr."""
        with caplog.at_level(logging.INFO, logger="fips.pipeline"):
            getattr(_Pipeline(), method)()

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
        assert _pipeline_messages(caplog)

    @pytest.mark.parametrize("method", ["get_inputs", "run"])
    def test_steps_are_below_warning(self, method, caplog):
        """The steps are INFO, so a logger left at WARNING shows none of them."""
        with caplog.at_level(logging.WARNING, logger="fips.pipeline"):
            getattr(_Pipeline(), method)()

        assert _pipeline_messages(caplog) == []
