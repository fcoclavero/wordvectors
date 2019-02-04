import pickle

import numpy as np

from functools import reduce

from gensim.models.keyedvectors import KeyedVectors


class KeyedVectorSingleton:
    """
    Utility class to manage single KeyedVector instances, as they are heavy to load.
    """
    __instance = None

    @staticmethod
    def getInstance(vector_path, vector_limit):
        """
        Static access method for singleton pattern.
        :return: the single KeyedVectorSingleton instance
        :type: KeyedVectorSingleton
        """
        if KeyedVectorSingleton.__instance is None: # if not created
            KeyedVectorSingleton.__instance = KeyedVectorSingleton(vector_path, vector_limit) # set the instance
        return KeyedVectorSingleton.__instance

    def __init__(self, vector_path, vector_limit):
        """
        Initialize constant parameters for all instances. Loads a gensim KeyedVectors object.
        :return: None
        """
        try:
            # loading a pickle is faster
            self.keyed_vectors = pickle.load(open(vector_path + ".pickle", 'rb'))
        except Exception as e:
            # if no pickle exists, load the object from the vec file and pickle it
            self.keyed_vectors = KeyedVectors.load_word2vec_format(vector_path, limit=vector_limit)
            pickle.dump(self.keyed_vectors, open(vector_path + ".pickle", 'wb'))

    @property
    def vector_dimensions(self):
        """
        Gensim does not provide a method the get the vector shape, so we must manually get it using a word in the corpus.
        :return: the vector dimensions
        """
        corpus_word = self.keyed_vectors.index2word[0] # get the first corpus word (any). NOTE: This function used to be index2entity
        corpus_word_vector = self.keyed_vectors[corpus_word]  # get its wordvector
        return corpus_word_vector.shape # return its shape

    @property
    def zero_vector(self):
        """
        Create a numpy zeros vector of the same shape as vectors in the corpus.
        :return: a zeros vector the same shape as those in the corpus
        :type: np.ndarray
        """
        return np.zeros(shape=self.vector_dimensions)

    def vector(self, word):
        """
        Get a word's embedding
        :param word: string with a single word.
        """
        return self.keyed_vectors[word]

    def doesnt_match(self, words):
        """
        Adapter method to simplify interface.
        """
        return self.keyed_vectors.doesnt_match(words)

    def most_similar_cosmul(self, positive, negative):
        """
        Adapter method to simplify interface.
        """
        return self.keyed_vectors.most_similar_cosmul(positive, negative)

    def wmdistance(self, term_1, term_2):
        """
        Adapter method to simplify interface.
        """
        return self.keyed_vectors.wmdistance(term_1, term_2)


class VectorFactory:
    """
    Factory class for creating word embedding vectors from any string.
    """
    def __init__(self, vector_path, vector_limit):
        """
        Creates a new word embedding vector factory.
        :return: a new VectorFactory
        :type: VectorFactory
        """
        self.keyed_vectors = KeyedVectorSingleton.getInstance(vector_path, vector_limit)

    @property
    def vector_dimensions(self):
        """
        Get the vector dimensions from the KeyedVectorSingleton object.
        :return: the vector dimensions
        """
        return self.keyed_vectors.vector_dimensions

    @property
    def zero_vector(self):
        """
        Get the zero vector from the KeyedVectorSingleton object.
        :return: the zero vector
        :type: np.ndarray
        """
        return self.keyed_vectors.zero_vector

    def _word_vector_partial_sum(self, partial_vector, word):
        """
        Private function used to reduce a tokenized string into a word embedding vector
        corresponding to the sum of all the vectors of the words which make up the string.
        Used for adding all the vectors of all vectors in a document via reduce.
        :param partial_vector: partial sum of vectors
        :type: np.ndarray
        :param word: the new word who's vector must be added to the partial sum vector
        :type: str
        :return: the result of adding the vector of word to partial_vector
        :type: np.ndarray
        """
        try:
            new_vector = self.word_vector(word)
        except Exception as e:
            new_vector = self.zero_vector

        return partial_vector + new_vector

    def word_vector(self, word):
        """
        Generates a word embedding vector which should encode semantic meaning.
        :param document: a single word
        :type: str
        :return: the sentence's word vector
        :type: np.ndarray
        """
        return self.keyed_vectors.vector(word)

    def document_vector(self, document):
        """
        Generates a document embedding vector which should encode its semantic meaning.
        This method carries out the most simple approach: averaging the vectors of
        the words that make up the sentence.
        :param document: a sentence
        :type: str
        :return: the sentence's word vector
        :type: np.ndarray
        """
        tokenized = document.split(" ")
        return reduce(self._word_vector_partial_sum, tokenized, self.zero_vector) / len(tokenized)
