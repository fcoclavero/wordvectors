__author__ = ["Francisco Clavero"]
__description__ = "Utility functions."
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

import sys

from random import randrange
from typing import Any

import numpy as np

from scipy.spatial.distance import cosine


def random_insert(lst: list, item: Any) -> None:
    """Inserts an item into a list, at a random position.

    Arguments:
        lst:
            The list to be extended.
        item:
            The item to be inserted.
    """
    lst.insert(randrange(len(lst) + 1), item)


def cosine_distance(vector_1: np.ndarray, vector_2: np.ndarray) -> float:
    """Computes the cosine distance between two vectors.

    Arguments:
        vector_1:
            A `numpy` vector.
        vector_2:
            Another `numpy` vector.

    Returns:
        The cosine distance between the two vectors.
    """
    if np.isnan(vector_1).any():
        print("vector_1 has nan")
    elif np.isnan(vector_1).any():
        print("vector_2 has nan")
    else:
        return cosine(vector_1, vector_2)
    return sys.float_info.max
