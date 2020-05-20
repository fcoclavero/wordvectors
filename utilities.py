__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


import sys

import numpy as np

from random import randrange
from scipy.spatial.distance import cosine


def random_insert(lst, item):
    """
    Inserts an item into a list, at a random position.
    :param lst: the list to be extended
    :type: List
    :param item: the item to be inserted
    :type: any
    :return: None
    """
    lst.insert(randrange(len(lst) + 1), item)


def distance_cosine(vector_1, vector_2):
    """
    Computes the cosine distance between two vectors.
    :param vector_1: a vector
    :type: np.ndarray
    :param vector_2: a vector
    :type: np.ndarray
    :return: the cosine distance between the two vectors.
    """
    if np.isnan(vector_1).any():
        print('vector_1 has nan')
    elif np.isnan(vector_1).any():
        print('vector_2 has nan')
    else:
        return cosine(vector_1, vector_2)
    return sys.float_info.max
