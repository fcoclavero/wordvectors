from .common import VectorFactory

from ..settings import EN


vector_factory = VectorFactory(EN['VECTOR_PATH'], EN['VECTOR_LIMIT'])


def word_vector(word):
    """
    Generates a word embedding vector which should encode semantic meaning.
    :param document: a single word
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