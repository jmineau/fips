"""
Defines the InversionPipeline class, which provides a structured workflow for inversion problems.
This class serves as a template for specific implementations, ensuring a consistent approach to loading data,
building covariance matrices, and running the inversion process.

The InversionPipeline class includes abstract methods that must be implemented by subclasses to handle
the specifics of data loading and covariance construction, while providing a default implementation for the overall workflow.
"""

from abc import ABC, abstractmethod
from typing import Any

from fips.covariance import CovarianceMatrix
from fips.estimators import Estimator
from fips.operators import ForwardOperator
from fips.problem import InverseProblem
from fips.vector import Vector


class InversionPipeline(ABC):
    """
    Blueprint for inversion.
    """

    def __init__(
        self,
        config: Any,
        problem: type[InverseProblem],
        estimator: type[Estimator] | str,
    ):
        self.config = config
        self.problem = problem
        self.estimator = estimator

    @abstractmethod
    def get_obs(self) -> Vector:
        pass

    @abstractmethod
    def get_prior(self) -> Vector:
        pass

    @abstractmethod
    def get_forward_operator(self, obs: Vector, prior: Vector) -> ForwardOperator:
        pass

    @abstractmethod
    def get_prior_error(self, prior: Vector) -> CovarianceMatrix:
        pass

    @abstractmethod
    def get_modeldata_mismatch(self, obs: Vector) -> CovarianceMatrix:
        pass

    def get_constant(self) -> Vector | None:
        return None

    def filter_state_space(self, obs: Vector, prior: Vector) -> tuple[Vector, Vector]:
        """Optional hook to align or trim the state space before building covariances."""
        return obs, prior

    def run(self, **kwargs) -> InverseProblem:
        """Executes the standard inversion workflow."""
        print("Loading problem inputs...")
        # Obs and prior are the core inputs that define the obs and state space.
        # They are used to build the forward operator and covariance matrices.
        obs = self.get_obs()
        prior = self.get_prior()

        # Optional filtering step to align state space (e.g. trim to observed subset)
        obs, prior = self.filter_state_space(obs=obs, prior=prior)

        # Build the forward operator and covariance matrices based on the obs and prior
        forward_operator = self.get_forward_operator(obs=obs, prior=prior)
        prior_error = self.get_prior_error(prior=prior)
        mdm = self.get_modeldata_mismatch(obs=obs)

        # Optional constants to be removed from the observations
        constant = self.get_constant()

        print("Initializing solver...")
        inversion = self.problem(
            obs=obs,
            prior=prior,
            forward_operator=forward_operator,
            prior_error=prior_error,
            modeldata_mismatch=mdm,
            constant=constant,
            **kwargs,
        )

        print("Solving...")
        inversion.solve(estimator=self.estimator)

        return inversion
