import numpy as np
from src.evaluation.metrics import recall_at_k, mean_average_precision

def test_metrics():
    gallery_classes = np.array([1,1,2,2])
    query_classes = np.array([1])
    results = np.array([[0,2,3,1]])
    assert recall_at_k(results, query_classes, gallery_classes, 2) == 0.5
    assert mean_average_precision(results, query_classes, gallery_classes) > 0
