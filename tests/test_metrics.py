import numpy as np

from src.metrics import mean_average_precision, recall_at_k


def test_metrics_multi_positive_query():
    gallery_labels = ["A", "B", "A", "C"]
    query_labels = ["A", "B"]
    similarity = np.array(
        [
            [0.9, 0.1, 0.8, 0.2],
            [0.2, 0.95, 0.1, 0.3],
        ]
    )

    assert recall_at_k(similarity, query_labels, gallery_labels, 1) == 1.0
    assert recall_at_k(similarity, query_labels, gallery_labels, 2) == 1.0
    assert mean_average_precision(similarity, query_labels, gallery_labels) == 1.0


def test_metrics_with_miss():
    gallery_labels = ["A", "B", "A", "C"]
    query_labels = ["C", "B"]
    similarity = np.array(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.2, 0.1, 0.05, 0.0],
        ]
    )

    recall_at_1 = recall_at_k(similarity, query_labels, gallery_labels, 1)
    recall_at_2 = recall_at_k(similarity, query_labels, gallery_labels, 2)
    assert recall_at_1 == 0.5
    assert recall_at_2 == 1.0
