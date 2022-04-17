__author__ = ["Francisco Clavero"]
__description__ = "Test suite for the `VectorFactory`."
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

from typing import List
from unittest import TestCase

import numpy as np

from ..es import VectorFactorySpanish
from ..settings import ES
from ..utilities import cosine_distance, random_insert


class VectorFactoryTestCase(TestCase):
    """Tests for creating vectors from strings and vector space properties."""

    VectorFactorySpanish

    @classmethod
    def setUpClass(cls) -> None:
        """Runs before tests. Loads word vectors and creates a new vector factory.

        See:
            https://docs.python.org/3.6/library/unittest.html
        """
        cls.keyed_vectors = VectorFactorySpanish.get_instance(
            ES["VECTOR_PATH"], ES["VECTOR_LIMIT"]
        )
        cls.word = "palabra"
        cls.word_vector = cls.keyed_vectors.word_vector(cls.word)
        cls.zero_vector = cls.keyed_vectors.zero_vector

    def test_singleton(self) -> None:
        """Test that only one `VectorFactory` instance is created."""
        new_keyed_vectors = VectorFactorySpanish.get_instance(
            ES["VECTOR_PATH"], ES["VECTOR_LIMIT"]
        )
        self.assertEqual(self.keyed_vectors, new_keyed_vectors)

    def test_dimensions(self) -> None:
        """Test that the loaded vectors have the correct dimensions."""
        self.assertEqual(self.keyed_vectors.vector_dimensions, (300,))

    def _word_vectors_similarity(
        self, positive: List[str], negative: List[str], required: str
    ) -> None:
        """Test similarities. Checks that required is the 10 most similar vectors to the
        result of the multiplicative combination objective proposed by Omer Levy and
        Yoav Goldberg.

        Arguments:
            positive:
                Words that contribute positively towards the similarity.
            negative:
                Words that contribute negatively towards the similarity.
            required:
                Word that must be present in the top 10 most similar vectors.
        """
        similar = self.keyed_vectors.most_similar_cosmul(
            positive=positive, negative=negative
        )  # (str, distance) pairs
        similar_strings = [s[0] for s in similar]  # get only strings
        self.assertTrue(required in similar_strings)

    def test_word_vectors_similarity(self) -> None:
        """Test simple similarities to verify that vectors have a correct distribution
        in the vector space; one that reflects the semantic meaning of the words.
        """
        self._word_vectors_similarity(
            positive=["rey", "mujer"], negative=["hombre"], required="reina"
        )
        self._word_vectors_similarity(
            positive=["jugar", "canta"], negative=["cantar"], required="juega"
        )
        self._word_vectors_similarity(
            positive=["pinochet", "argentino"], negative=["chileno"], required="perón"
        )
        self._word_vectors_similarity(
            positive=["santiago", "venezuela"], negative=["chile"], required="caracas"
        )

    def _word_vectors_match(self, similar: List[str], different: str) -> None:
        """Test matching. Check that when all the words in "similar" plus the
        "different" word are matched, "different" is correctly identified as the most
        dissimilar.

        Arguments:
            similar:
                List of similar words.
            different:
                A word semantically different from the words in "similar".
        """
        words = similar
        random_insert(words, different)
        prediction = self.keyed_vectors.doesnt_match(words)
        self.assertEquals(prediction, different)

    def test_word_vectors_match(self) -> None:
        """Test simple matching to verify that vectors have a correct distribution in
        the vector space; one that reflects the semantic meaning of the words.
        """
        self._word_vectors_match(similar=["blanco", "azul", "rojo"], different="chile")
        self._word_vectors_match(
            similar=["lunes", "martes", "miercoles"], different="septiembre"
        )

    def test_word_vector_sum(self) -> None:
        """Test the adding a word vector with a string (which is vectorized by the
        function).
        """
        # If the partial vector is zero, the word's vector should be returned
        np.testing.assert_array_equal(
            self.word_vector,
            self.keyed_vectors._word_vector_partial_sum(self.zero_vector, self.word),
        )
        # If a non-existent word is added, the result should be the partial vector
        np.testing.assert_array_equal(
            self.zero_vector,
            self.keyed_vectors._word_vector_partial_sum(self.zero_vector, "plbra"),
        )
        # Simple sum
        np.testing.assert_array_equal(
            self.word_vector + self.word_vector,
            self.keyed_vectors._word_vector_partial_sum(self.word_vector, self.word),
        )

    def test_vector_from_word(self) -> None:
        """Simplest test. Check if the vector created for a word is the same vector."""
        np.testing.assert_array_equal(
            self.word_vector, self.keyed_vectors.word_vector(self.word)
        )

    def _document_vector(self, similar_1: str, similar_2: str, different: str) -> None:
        """Test that semantically similar documents are correctly closer to each other
        than to a semantically different document. The generated word embeddings for the
        documents must reflect this relation in the vector space.

        Arguments:
            similar_1:
                A document that is semantically similar to `similar_2`.
            similar_2:
                A document that is semantically similar to `similar_1`.
            different:
                A document that is semantically different to both `similar_1` and
                `similar_2`.
        """
        # First, we must check that the similar sentences are more similar to each other
        # than to the different sentence. We will first use the document distance
        # function provided by `gensim`.
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
        distance_1 = cosine_distance(similar_vector_1, similar_vector_2)
        distance_2 = cosine_distance(similar_vector_1, different_vector)
        distance_3 = cosine_distance(similar_vector_2, different_vector)

        self.assertGreaterEqual(distance_2, distance_1)
        self.assertGreaterEqual(distance_3, distance_1)

    def test_document_vector(self) -> None:
        """Test the complete word vector creation.

        As different sentence vector creation methods can be implemented, their position
        in the vector space, which must reflect the semantic meaning of the sentence,
        must be tested.
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
