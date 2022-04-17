__author__ = ["Francisco Clavero"]
__description__ = "Project settings and constants."
__email__ = ["fcoclavero32@gmail.com"]
__status__ = "Prototype"

import os

from typing import Dict, Optional

from dotenv import load_dotenv


# Load env
load_dotenv()

EMBEDDINGS_DIR: str = os.environ["EMBEDDINGS_DIR"]
"""Path to the root embedding directory."""

EN: Dict[str, Optional[int]] = {
    "VECTOR_PATH": os.path.join(EMBEDDINGS_DIR, "en", "wiki-news-300d-1M-subword.vec"),
    "VECTOR_LIMIT": None,
}
"""English `VectorFactory` settings."""

ES: Dict[str, Optional[int]] = {
    "VECTOR_PATH": os.path.join(EMBEDDINGS_DIR, "es", "FastText_SBWC.vec"),
    "VECTOR_LIMIT": None,
}
"""Spanish `VectorFactory` settings."""
