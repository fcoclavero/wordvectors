__author__ = ["Francisco Clavero"]
__description__ = "`VectorFactory` for spanish."
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

from typing import Optional

import numpy as np

from .settings import ES
from .vector_factory import VectorFactory


class VectorFactorySpanish(VectorFactory):
    """Utility class to manage a single `VectorFactory` instance for this language, as
    they are heavy to load.
    """

    __instance: "VectorFactorySpanish" = None
    """The singleton instance."""

    @staticmethod
    def get_instance(
        vector_path: str, vector_limit: Optional[int]
    ) -> "VectorFactorySpanish":
        """Static access method for the singleton pattern.

        Arguments:
            vector_path:
                The file path to the saved `word2vec` format file.
            vector_limit:
                Maximum number of word-vectors to read from the file. The default,
                `None`, means read all.

        Returns:
            The single `VectorFactorySpanish` instance.
        """
        if VectorFactorySpanish.__instance is None:  # if not created
            VectorFactorySpanish.__instance = VectorFactorySpanish(
                vector_path, vector_limit
            )  # set the instance
        return VectorFactorySpanish.__instance


vector_factory = VectorFactorySpanish.get_instance(
    ES["VECTOR_PATH"], ES["VECTOR_LIMIT"]
)


def word_vector(word: str) -> np.ndarray:
    """Generates a word embedding vector which should encode its semantic meaning.

    Arguments:
        word:
            A single word.

    Returns:
        The word's vector.
    """
    return vector_factory.word_vector(word)


def document_vector(document: str) -> np.ndarray:
    """Generates a document embedding vector which should encode its semantic meaning.

    This method carries out the most simple approach: averaging the vectors of the words
    that make up the sentence.

    Arguments:
        A document string.

    Returns:
        The document's vector.
    """
    return vector_factory.document_vector(document)


def zero_vector() -> np.ndarray:
    """Returns a zero vector with the same shapes as word and document vectors.

    Returns:
        The zero vector.
    """
    return vector_factory.zero_vector
