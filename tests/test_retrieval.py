import numpy as np
from src.retrieval.exact import ExactCosineRetriever

def test_exact_identity():
    x = np.eye(3, dtype="float32")
    r = ExactCosineRetriever(x)
    scores, ids = r.search(x, 1)
    assert np.array_equal(ids.ravel(), np.arange(3))
