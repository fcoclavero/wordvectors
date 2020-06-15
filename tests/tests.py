__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


from unittest import TestCase

import numpy as np

from ..es import VectorFactorySpanish
from ..settings import ES
from ..utilities import distance_cosine
from ..utilities import random_insert


class VectorFactoryTestCase(TestCase):
    """
    Tests for creating vectors from strings and vector space properties.
    """

    @classmethod
    def setUpClass(cls):
        """
        Runs before tests: https://docs.python.org/3.6/library/unittest.html
        Loads word vectors and creates a new vector factory
        :return: None
        """
        cls.keyed_vectors = VectorFactorySpanish.getInstance(ES["VECTOR_PATH"], ES["VECTOR_LIMIT"])
        cls.word = "palabra"
        cls.word_vector = cls.keyed_vectors.word_vector(cls.word)
        cls.zero_vector = cls.keyed_vectors.zero_vector

    def test_singleton(self):
        """
        Test that only one instance is created.
        :return: None
        """
        new_keyed_vectors = VectorFactorySpanish.getInstance(ES["VECTOR_PATH"], ES["VECTOR_LIMIT"])
        self.assertEqual(self.keyed_vectors, new_keyed_vectors)

    def test_dimensions(self):
        """
        Test that the loaded vectors have the correct dimensions.
        :return: None
        """
        self.assertEqual(self.keyed_vectors.vector_dimensions, (300,))

    def _word_vectors_similarity(self, positive, negative, required):
        """
        Test similarities. Checks that required is the 10 most similar vectors to the
        result of the multiplicative combination objective proposed by Omer Levy and
        Yoav Goldberg.
        :param positive: words that contribute positively towards the similarity
        :type: List[str]
        :param negative: words that contribute negatively towards the similarity
        :type: List[str]
        :param required: word that must be present in the top 10 most similar vectors
        :type: str
        :return: None
        """
        similar = self.keyed_vectors.most_similar_cosmul(positive=positive, negative=negative)  # (str, distance) pairs
        similar_strings = [s[0] for s in similar]  # get only strings
        self.assertTrue(required in similar_strings)

    def test_word_vectors_similarity(self):
        """
        Test simple similarities to verify that vectors have a correct distribution in
        the vector space; one that reflects the semantic meaning of the words.
        :return: None
        """
        self._word_vectors_similarity(positive=["rey", "mujer"], negative=["hombre"], required="reina")
        self._word_vectors_similarity(positive=["jugar", "canta"], negative=["cantar"], required="juega")
        self._word_vectors_similarity(positive=["pinochet", "argentino"], negative=["chileno"], required="perón")
        self._word_vectors_similarity(positive=["santiago", "venezuela"], negative=["chile"], required="caracas")

    def _word_vectors_match(self, similar, different):
        """
        Test matching. Check that when all the words in "similar" plus the "different"
        word are matched, "different" is correctly identified as the most dissimilar.
        :param similar: list of similar words
        :type: List[str]
        :param different: a word semantically different from the words in "similar"
        :type: str
        :return: None
        """
        words = similar
        random_insert(words, different)
        prediction = self.keyed_vectors.doesnt_match(words)
        self.assertEquals(prediction, different)

    def test_word_vectors_match(self):
        """
        Test simple matching to verify that vectors have a correct distribution in
        the vector space; one that reflects the semantic meaning of the words.
        :return: None
        """
        self._word_vectors_match(similar=["blanco", "azul", "rojo"], different="chile")
        self._word_vectors_match(similar=["lunes", "martes", "miercoles"], different="septiembre")

    def test_word_vector_sum(self):
        """
        Test the adding a word vector with a string (which is vectorized by the function)
        :return: None
        """
        # If the partial vector is zero, the word's vector should be returned
        np.testing.assert_array_equal(
            self.word_vector, self.keyed_vectors._word_vector_partial_sum(self.zero_vector, self.word)
        )

        # If a non-existent word is added, the result should be the partial vector
        np.testing.assert_array_equal(
            self.zero_vector, self.keyed_vectors._word_vector_partial_sum(self.zero_vector, "plbra")
        )

        # Simple sum
        np.testing.assert_array_equal(
            self.word_vector + self.word_vector,
            self.keyed_vectors._word_vector_partial_sum(self.word_vector, self.word),
        )

    def test_vector_from_word(self):
        """
        Simplest test. Check if the vector created for a word is the same vector.
        :return: None
        """
        np.testing.assert_array_equal(self.word_vector, self.keyed_vectors.word_vector(self.word))

    def _document_vector(self, similar_1, similar_2, different):
        """
        Test that semantically similar documents are correctly closer to each other than
        to a semantically different document. The generated word embeddings for the
        documents must reflect this relation in the vector space.
        :param similar_1: a document that is semantically similar to similar_2
        :type: str
        :param similar_2: a document that is semantically similar to similar_2
        :type: str
        :param different: a document that is semantically different to similar_1
        and similar_2
        :type: str
        :return: None
        """
        # First, we must check that the similar sentences are more similar to each other
        # than to the different sentence. We will first use the document distance
        # function provided by genism.
        distance_1 = self.keyed_vectors.wmdistance(similar_1, similar_2)
        distance_2 = self.keyed_vectors.wmdistance(similar_1, different)
        distance_3 = self.keyed_vectors.wmdistance(similar_2, different)

        self.assertGreaterEqual(distance_2, distance_1)
        self.assertGreaterEqual(distance_3, distance_1)

        # Vectorize the sentences
        similar_vector_1 = self.keyed_vectors.document_vector(similar_1)
        similar_vector_2 = self.keyed_vectors.document_vector(similar_2)
        different_vector = self.keyed_vectors.document_vector(different)

        # Now we check if the generated vectors maintain the relation above
        distance_1 = distance_cosine(similar_vector_1, similar_vector_2)
        distance_2 = distance_cosine(similar_vector_1, different_vector)
        distance_3 = distance_cosine(similar_vector_2, different_vector)

        self.assertGreaterEqual(distance_2, distance_1)
        self.assertGreaterEqual(distance_3, distance_1)

    def test_document_vector(self):
        """
        Test the complete word vector creation. As different sentence vector creation
        methods can be implemented, their position in the vector space, which must
        reflect the semantic meaning of the sentence, must be tested.
        :return: None
        """
        self._document_vector(
            similar_1="trayectoria",  # net_cod = [5]
            similar_2="historial trayectoria",  # net_cod = [5]
            different="atencion",  # net_cod = [1]
        )
        self._document_vector(
            similar_1="costo corriente alto costo gasto mensual",  # net_cod = [61]
            similar_2="costo mantencion costo alto mantencion",  # net_cod = [61]
            different="precio variedad servicio",  # net_cod = [8,2]
        )
        self._document_vector(
            similar_1="sucursales respuesta fluidez bancaria",  # net_cod = [12,5,7]
            similar_2="respuesta rapida",  # net_cod = [7]
            different="menores costos",  # net_cod = [57]
        )
