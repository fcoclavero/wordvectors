__author__ = ["Francisco Clavero"]
__description__ = "Base `VectorFactory` class."
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

import pickle

from functools import reduce
from typing import List, Optional, Tuple

import numpy as np

from gensim.models.keyedvectors import KeyedVectors


class VectorFactory:
    """Factory class for creating word embedding vectors from any string."""

    def __init__(self, vector_path: str, vector_limit: Optional[int]) -> None:
        """Initialize factory and load a `gensim` `KeyedVectors` object.

        Tries to load from pickle, otherwise loads `word2vec` and pickles. Loading from
        pickle is much faster.

        Arguments:
            vector_path:
                The file path to the saved `word2vec` format file.
            vector_limit:
                Maximum number of word-vectors to read from the file. The default,
                `None`, means read all.
        """
        try:
            # loading a pickle is faster
            self.keyed_vectors = pickle.load(open(vector_path + ".pickle", "rb"))
        except FileNotFoundError:
            # if no pickle exists, load the object from the vec file and pickle it
            self.keyed_vectors = KeyedVectors.load_word2vec_format(
                vector_path, limit=vector_limit
            )
            pickle.dump(self.keyed_vectors, open(vector_path + ".pickle", "wb"))

    @property
    def vector_dimensions(self) -> Tuple[int]:
        """Gensim does not provide a method the get the vector shape, so we must
        manually get it using a word in the corpus.
        """
        # Get the first corpus word (any).
        corpus_word = self.keyed_vectors.index2word[
            0
        ]  # NOTE: This function used to be index2entity
        corpus_word_vector = self.keyed_vectors[corpus_word]  # get its word vector
        return corpus_word_vector.shape  # return its shape

    @property
    def zero_vector(self) -> np.ndarray:
        """Numpy zeros vector of the same shape as vectors in the corpus."""
        return np.zeros(shape=self.vector_dimensions)

    def doesnt_match(self, words: List[str]) -> str:
        """Adapter method to simplify interface.

        See:
            https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.doesnt_match
        """
        return self.keyed_vectors.doesnt_match(words)

    def most_similar_cosmul(
        self,
        positive: Optional[List[str]],
        negative: Optional[List[str]],
        topn: Optional[int] = 10,
    ):
        """Adapter method to simplify interface.

        See:
            https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.most_similar_cosmul
        """
        return self.keyed_vectors.most_similar_cosmul(positive, negative, topn)

    def wmdistance(
        self, document_1: List[str], document_2: List[str], norm: bool = True
    ):
        """Adapter method to simplify interface.

        See:
            https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.KeyedVectors.wmdistance
        """
        return self.keyed_vectors.wmdistance(document_1, document_2)

    def _word_vector_partial_sum(
        self, partial_vector: np.ndarray, word: str
    ) -> np.ndarray:
        """Private function used to reduce a tokenized string into a word embedding
        vector corresponding to the sum of all the vectors of the words which make up
        the string.

        Used for adding all the vectors of all vectors in a document via `reduce`.

        Arguments:
            partial_vector:
                Partial sum of vectors.
            word:
                The new word whose vector must be added to the partial sum vector.

        Returns:
            The result of adding the vector of `word` to `partial_vector`.
        """
        try:
            new_vector = self.word_vector(word)
        except KeyError:
            new_vector = self.zero_vector

        return partial_vector + new_vector

    def word_vector(self, word: str) -> np.ndarray:
        """Generates a word embedding vector which should encode semantic meaning.

        Arguments:
            word:
                A single word.

        Returns:
            The word's vector.
        """
        return self.keyed_vectors.get_vector(word)

    def document_vector(self, document: str) -> np.ndarray:
        """Generates a document embedding vector which should encode its semantic
        meaning.

        This method carries out the most simple approach: averaging the vectors of the
        words that make up the sentence.

        Arguments:
            A document string.

        Returns:
            The document's vector.
        """
        tokenized = document.split(" ")
        return reduce(self._word_vector_partial_sum, tokenized, self.zero_vector) / len(
            tokenized
        )
