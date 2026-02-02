import numpy as np
import pandas as pd

from fips.indices import sanitize_index
from fips.structures import Matrix, Vector


class ForwardOperator(Matrix):
    """
    Represents the Jacobian / Forward Operator (H).
    Columns = State Space, Rows = Obs Space.
    """

    @property
    def state_index(self) -> pd.Index:
        return self.columns

    @property
    def obs_index(self) -> pd.Index:
        return self.index

    def convolve(
        self, state: Vector | pd.Series | np.ndarray, float_precision: int | None = None
    ) -> pd.Series:
        """Convolve (project) a state vector through the forward operator."""
        if isinstance(state, Vector):
            s = state.data.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, pd.Series):
            s = state.copy()
            s.index = sanitize_index(s.index, float_precision)
        elif isinstance(state, np.ndarray):
            if state.shape[0] != self.data.shape[1]:
                raise ValueError(
                    f"Shape mismatch: Operator expects {self.data.shape[1]}, got {state.shape[0]}"
                )
            x_vals = state
            return pd.Series(
                self.data.values @ x_vals, index=self.obs_index, name="modeled_obs"
            )
        else:
            raise TypeError("State must be a Vector, pandas Series, or numpy array.")

        x_vals = s.reindex(self.state_index).fillna(0.0).values
        y_values = self.data.values @ x_vals
        return pd.Series(y_values, index=self.obs_index, name=f"{s.name}_obs")


def convolve(
    state: Vector | pd.Series | np.ndarray,
    forward_operator: ForwardOperator | pd.DataFrame,
    float_precision: int | None = None,
) -> pd.Series:
    """Helper to convolve a state vector with a forward operator matrix."""
    if isinstance(forward_operator, pd.DataFrame):
        forward_operator = ForwardOperator(forward_operator)

    return forward_operator.convolve(state=state, float_precision=float_precision)
