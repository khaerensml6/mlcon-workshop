"""
This component generates a set of initial prompts that will be used to retrieve images
from the LAION-5B dataset.
"""
import itertools
import logging
import typing as t
import io
import tempfile

import dask.dataframe as dd
import pandas as pd
from PIL import Image

from fondant.component import PandasTransformComponent

logger = logging.getLogger(__name__)


class ConditioningComponent(PandasTransformComponent):
    def __init__(self, *args) -> None:
        """
        Initialization of the component
        """
        pass

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Implement your custom logic in this single method
        Args:
            dataframe: A Pandas dataframe containing the data
        Returns:
            A pandas dataframe containing the transformed data
        """

        conditionings = []

        for image_data in dataframe["images"]["data"]:
            temp_file = tempfile.NamedTemporaryFile()

            with open(temp_file.name, "wb") as f:
                f.write(image_data)
            image = Image.open(temp_file.name)

            ### TODO: process images here to create conditioning images ###

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            conditionings.append(img_byte_arr)

        dataframe.columns = [["conditionings"], ["data"]]
        dataframe["conditionings"]["data"] = conditionings

        return dataframe
