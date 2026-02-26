"""
Defines the InversionPipeline class, which provides a structured workflow for inversion problems.
This class serves as a template for specific implementations, ensuring a consistent approach to loading data,
building covariance matrices, and running the inversion process.

The InversionPipeline class includes abstract methods that must be implemented by subclasses to handle
the specifics of data loading and covariance construction, while providing a default implementation for the overall workflow.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from fips.covariance import CovarianceMatrix
from fips.estimators import Estimator
from fips.operators import ForwardOperator
from fips.problem import InverseProblem
from fips.vector import Vector

_Problem = TypeVar("_Problem", bound=InverseProblem)


class InversionPipeline(ABC, Generic[_Problem]):
    """
    Blueprint for inversion.
    """

    def __init__(
        self,
        config: Any,
        problem: type[_Problem],
        estimator: type[Estimator] | str,
    ):
        self.config = config
        self._InverseProblem = problem
        self.estimator = estimator

    @abstractmethod
    def get_obs(self) -> Vector:
        pass

    @abstractmethod
    def get_prior(self) -> Vector:
        pass

    def filter_state_space(self, obs: Vector, prior: Vector) -> tuple[Vector, Vector]:
        """Optional hook to align or trim the state space before building covariances."""
        return obs, prior

    @abstractmethod
    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        pass

    @abstractmethod
    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        pass

    @abstractmethod
    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        pass

    def get_constant(self, obs: Vector) -> Vector | None:
        return None

    def aggregate_obs_space(
        self,
        obs: Vector,
        forward_operator: ForwardOperator,
        modeldata_mismatch: CovarianceMatrix,
        constant: Vector | None,
    ) -> tuple[Vector, ForwardOperator, CovarianceMatrix, Vector | None]:
        """Optional hook to aggregate the observation space (e.g. hourly → daily)."""
        return obs, forward_operator, modeldata_mismatch, constant

    def get_inputs(self) -> dict[str, Any]:
        step_start = time.perf_counter()
        # Obs and prior are the core inputs that define the obs and state space.
        # They are used to build the forward operator and covariance matrices.
        print("Loading observations...")
        obs = self.get_obs()
        print(f"Observations loaded in {time.perf_counter() - step_start:.2f}s")

        step_start = time.perf_counter()
        print("Loading prior...")
        prior = self.get_prior()
        print(f"Prior loaded in {time.perf_counter() - step_start:.2f}s")

        # Optional filtering step to align state space (e.g. trim to observed subset)
        step_start = time.perf_counter()
        print("Optionally, filtering state space...")
        obs, prior = self.filter_state_space(obs=obs, prior=prior)
        print(
            "Optionally, state space filtered in "
            f"{time.perf_counter() - step_start:.2f}s"
        )

        # Build the forward operator and covariance matrices based on the obs and prior
        step_start = time.perf_counter()
        print("Building forward operator...")
        forward_operator = self.get_forward_operator(obs=obs, prior=prior)
        print(f"Forward operator built in {time.perf_counter() - step_start:.2f}s")

        step_start = time.perf_counter()
        print("Building prior error covariance...")
        prior_error = self.get_prior_error(prior=prior)
        print(
            f"Prior error covariance built in {time.perf_counter() - step_start:.2f}s"
        )

        step_start = time.perf_counter()
        print("Building model-data mismatch covariance...")
        mdm = self.get_modeldata_mismatch(obs=obs)
        print(
            f"Model-data mismatch covariance built in {time.perf_counter() - step_start:.2f}s"
        )

        # Optional constants to be removed from the observations
        step_start = time.perf_counter()
        print("Loading constant term (if any)...")
        constant = self.get_constant(obs=obs)
        print(f"Constant term loaded in {time.perf_counter() - step_start:.2f}s")

        # Optional obs-space aggregation (e.g. hourly → daily)
        step_start = time.perf_counter()
        print("Optionally, aggregating observation space (if needed)...")
        obs, forward_operator, mdm, constant = self.aggregate_obs_space(
            obs=obs,
            forward_operator=forward_operator,
            modeldata_mismatch=mdm,
            constant=constant,
        )
        print(
            "Optionally, observation space aggregated in "
            f"{time.perf_counter() - step_start:.2f}s"
        )
        return dict(
            obs=obs,
            prior=prior,
            forward_operator=forward_operator,
            prior_error=prior_error,
            modeldata_mismatch=mdm,
            constant=constant,
        )

    def run(self, **kwargs) -> _Problem:
        """Executes the standard inversion workflow."""
        total_start = time.perf_counter()
        print("Getting problem inputs...")
        inputs = self.get_inputs()
        print(f"Inputs prepared in {time.perf_counter() - total_start:.2f}s")

        step_start = time.perf_counter()
        print("Initializing solver...")
        self.problem = self._InverseProblem(
            **inputs,
            **kwargs,
        )
        print(f"Solver initialized in {time.perf_counter() - step_start:.2f}s")

        step_start = time.perf_counter()
        print("Solving...")
        self.problem.solve(estimator=self.estimator)
        print(f"Solve completed in {time.perf_counter() - step_start:.2f}s")

        print(f"Total pipeline time: {time.perf_counter() - total_start:.2f}s")

        return self.problem
