from typing import Iterable, List

import numpy as np


def recall_at_k(
    similarity: np.ndarray,
    query_labels: List[str],
    gallery_labels: List[str],
    k: int,
) -> float:
    if similarity.shape[0] == 0:
        return 0.0
    k = min(k, similarity.shape[1])
    topk_idx = np.argsort(-similarity, axis=1)[:, :k]
    hits = 0
    for i, indices in enumerate(topk_idx):
        label = query_labels[i]
        if any(gallery_labels[j] == label for j in indices):
            hits += 1
    return hits / similarity.shape[0]


def average_precision(
    similarity_row: np.ndarray, query_label: str, gallery_labels: List[str]
) -> float:
    order = np.argsort(-similarity_row)
    positives = [1 if gallery_labels[idx] == query_label else 0 for idx in order]
    if sum(positives) == 0:
        return 0.0
    precision_sum = 0.0
    hit_count = 0
    for rank, is_pos in enumerate(positives, start=1):
        if is_pos:
            hit_count += 1
            precision_sum += hit_count / rank
    return precision_sum / hit_count


def mean_average_precision(
    similarity: np.ndarray, query_labels: List[str], gallery_labels: List[str]
) -> float:
    if similarity.shape[0] == 0:
        return 0.0
    aps = [
        average_precision(similarity[i], query_labels[i], gallery_labels)
        for i in range(similarity.shape[0])
    ]
    return float(np.mean(aps))
