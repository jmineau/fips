from collections.abc import Callable

import pandas as pd


class ObsAggregator:
    """
    A class for grouping and aggregating pandas Series or DataFrames with reusable configuration.
    """

    def __init__(
        self,
        groupers: str | list[str] | dict[str, pd.Grouper],
        method: str | Callable | dict[str, str | Callable],
    ):
        """
        Initialize the Aggregator.

        Parameters
        ----------
        groupers : str, list of str, or dict mapping column names to pd.Grouper objects.

        method : str, callable, or dict mapping column names to aggregation methods.
            The aggregation method to apply to each group.
        """
        self.groupers = groupers
        self.method = method

    def aggregate(self, data: pd.Series) -> pd.Series:
        """
        Apply the configured aggregation to the provided series.

        Parameters
        ----------
        data : pd.Series
            The data to aggregate.

        Returns
        -------
        pd.Series
            The aggregated data.
        """
        grouped = data.groupby(self.groupers)
        return grouped.agg(self.method)

    def __call__(self, data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Apply the aggregation to a Data object.

        Parameters
        ----------
        data : pd.Series | pd.DataFrame
            The data to aggregate.

        Returns
        -------
        pd.Series | pd.DataFrame
            A new object with aggregated data.
        """
        assert isinstance(data, (pd.Series, pd.DataFrame)), (
            "Data must be a pandas Series or DataFrame."
        )
        if not isinstance(data, pd.Series):
            return data.groupby(self.groupers).agg(self.method)

        return self.aggregate(data)
