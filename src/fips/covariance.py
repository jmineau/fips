import logging

import numpy as np
import pandas as pd

from fips.matrix import SymmetricMatrix

logger = logging.getLogger(__name__)


class CovarianceMatrix(SymmetricMatrix):
    """
    Represents a symmetric Covariance Matrix.

    Covariance matrices are used to represent error covariances in the inversion framework.
    They can be constructed from variances and correlation matrices.
    """

    def get_variances(self, block: str | None = None) -> pd.Series:
        """
        Get the variances (diagonal elements) of the covariance matrix.

        Parameters
        ----------
        block : str, optional
            If specified, return variances only for the given block.

        Returns
        -------
        pd.Series
            Series of variances indexed by state vector index.
        """
        variances = pd.Series(
            np.diag(self.data.values), index=self.data.index, name="variance"
        )
        if block is not None:
            variances = variances.xs(block, level="block")
        return variances
