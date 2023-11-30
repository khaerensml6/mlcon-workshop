"""
This component generates a set of initial prompts that will be used to retrieve images
from the LAION-5B dataset.
"""
import itertools
import logging
import typing as t

import dask.dataframe as dd
import pandas as pd

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class FilterComponent(PandasTransformComponent):
    def __init__(self, *args, max_length: t.Optional[int]) -> None:
        """
        Generate a set of initial prompts that will be used to retrieve images from the
        LAION-5B dataset.

        Args:
            n_rows_to_load: Optional argument that defines the number of rows to load.
                Useful for testing pipeline runs on a small scale
        """
        self.max_length = max_length

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Implement your custom logic in this single method
        Args:
            dataframe: A Pandas dataframe containing the data
        Returns:
            A pandas dataframe containing the transformed data
        """

        dataframe = dataframe[dataframe["captions"]["text"].str.len() <= self.max_length]

        ### TODO: Add extra filtering logic here ###

        return dataframe
