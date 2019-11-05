__author__ = ['Francisco Clavero']
__email__ = ['fcoclavero32@gmail.com']
__status__ = 'Prototype'


import os

from dotenv import load_dotenv


# Load env
load_dotenv()

EMBEDDINGS_DIR = os.environ['EMBEDDINGS_DIR']

EN = {
    'VECTOR_PATH': os.path.join(EMBEDDINGS_DIR, 'en', 'wiki-news-300d-1M-subword.vec'),
    'VECTOR_LIMIT': None
}

ES = {
    'VECTOR_PATH': os.path.join(EMBEDDINGS_DIR, 'es', 'FastText_SBWC.vec'),
    'VECTOR_LIMIT': None
}