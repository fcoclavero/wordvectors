__author__ = ["Francisco Clavero"]
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"


from .settings import EN
from .vector_factory import VectorFactory


class VectorFactoryEnglish(VectorFactory):
    """
    Utility class to manage a single VectorFactory instance for this language, as they are heavy to load.
    """

    __instance = None

    @staticmethod
    def getInstance(vector_path, vector_limit):
        """
        Static access method for singleton pattern.
        :return: the single KeyedVectorSingleton instance
        :type: KeyedVectorSingleton
        """
        if VectorFactoryEnglish.__instance is None:  # if not created
            VectorFactoryEnglish.__instance = VectorFactoryEnglish(vector_path, vector_limit)  # set the instance
        return VectorFactoryEnglish.__instance

    def __init__(self, vector_path, vector_limit):
        super().__init__(vector_path, vector_limit)


vector_factory = VectorFactoryEnglish.getInstance(EN["VECTOR_PATH"], EN["VECTOR_LIMIT"])


def word_vector(word):
    """
    Generates a word embedding vector which should encode semantic meaning.
    :param word: a single word
    :type: str
    :return: the sentence's word vector
    :type: np.ndarray
    """
    return vector_factory.word_vector(word)


def document_vector(document):
    """
    Generates a document embedding vector which should encode its semantic meaning.
    This method carries out the most simple approach: averaging the vectors of
    the words that make up the sentence.
    :param document: a sentence
    :type: str
    :return: the sentence's word vector
    :type: np.ndarray
    """
    return vector_factory.document_vector(document)


def zero_vector():
    """
    Returns a zero vector with the same shapes as word and document vectors.
    :return: the zero vector
    :type: np.ndarray
    """
    return vector_factory.zero_vector
