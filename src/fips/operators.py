import numpy as np
import pandas as pd

from fips.matrix import Matrix
from fips.vector import Vector, VectorLike


class ForwardOperator(Matrix):
    """
    Forward operator matrix mapping state vectors to observation space.

    A ForwardOperator wraps a pandas DataFrame and provides methods
    to convolve state vectors through the operator to produce modeled observations.

    The foward operator, or Jacobian matrix, is a key component of inverse problems.
    It defines how changes in the state vector affect the observations.
    The rows correspond to observations and the columns to state variables.
    """

    @property
    def state_index(self) -> pd.Index:
        return self.columns

    @property
    def obs_index(self) -> pd.Index:
        return self.index

    def convolve(
        self,
        state: VectorLike,
        round_index: int | None = None,
        verify_overlap: bool = True,
    ) -> pd.Series:
        """Convolve a state vector through the forward operator."""
        state = Vector(state)

        if round_index:
            op = self.round_index(round_index, axis="both")
            state = state.round_index(round_index)
        else:
            op = self

        state = state.reindex(
            op.state_index, fill_value=0.0, verify_overlap=verify_overlap
        )

        x_vals = state.values
        y_values = op.data.values @ x_vals
        name = f"{state.name}_obs" if state.name else None
        return pd.Series(y_values, index=op.obs_index, name=name)


def convolve(
    state: Vector | pd.Series | np.ndarray,
    forward_operator: ForwardOperator | pd.DataFrame,
    round_index: int | None = None,
    verify_overlap: bool = True,
) -> pd.Series:
    """Helper to convolve a state vector with a forward operator matrix."""
    forward_operator = ForwardOperator(forward_operator)

    return forward_operator.convolve(
        state=state, round_index=round_index, verify_overlap=verify_overlap
    )
