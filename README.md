# wordvectors

Utility functions for creating word and document embeddings.

The package currently supports both english and spanish, and provides the functions `word_vector` and `document_vector` to create word vectors from single-word strings and multi-word strings, respectively.

Example usage:

```python
from wordvectors.en import document_vector, word_vector


word_vector('dog')
document_vector('this is my little gray dog')
```