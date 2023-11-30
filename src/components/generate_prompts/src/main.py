"""
This component generates a set of initial prompts that will be used to retrieve images
from the LAION-5B dataset.
"""
import itertools
import logging
import typing as t
import random

import dask.dataframe as dd
import pandas as pd

from fondant.component import DaskLoadComponent

logger = logging.getLogger(__name__)

animals = [
    "a dog",
    "a cat",
    "a horse",
    "a peacock",
    "a tucan",
    "a sheep",
    "a rabbit",
    "a chicken",
    "a duck",
    "an ostrich",
    "a mongoose",
    "a platypus",
    "a dolphin",
    "a whale",
    "a shark",
    "a turtle",
    "a crocodile",
    "a giraffe",
    "an elephant",
]


clothing = [
    "a shirt",
    "pants",
    "a hat",
    "shoes",
    "a hoody",
    "a jacket",
    "boots",
]

media = [
    "photograph",
    "painting",
    "drawing",
    "render",
    "sketch",
    "etching",
    "illustration",
    "digital painting",
    "digital drawing",
    "digital illustration",
    "digital render",
]


def make_seed_prompt(animal: str, clothing_article: str, medium: str) -> str:
    """Generate a seed prompt to query LAION."""

    prompt = f"a {medium} of {animal} wearing {clothing_article}"
    return prompt


class GeneratePromptsComponent(DaskLoadComponent):
    def __init__(self, *args, n_rows_to_load: t.Optional[int]) -> None:
        """
        Generate a set of initial prompts that will be used to retrieve images from the
        LAION-5B dataset.

        Args:
            n_rows_to_load: Optional argument that defines the number of rows to load.
                Useful for testing pipeline runs on a small scale
        """
        self.n_rows_to_load = n_rows_to_load

    def load(self) -> dd.DataFrame:

        room_tuples = list(itertools.product(animals, clothing, media))
        random.shuffle(room_tuples)

        prompts = map(lambda x: make_seed_prompt(*x), room_tuples)

        pandas_df = pd.DataFrame(prompts, columns=["prompts_text"])

        if self.n_rows_to_load:
            pandas_df = pandas_df.head(self.n_rows_to_load)

        df = dd.from_pandas(pandas_df, npartitions=1)
        df.index = df.index.astype(str)
        return df
