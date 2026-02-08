import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import load_embeddings
from src.index import cosine_similarity


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), p))


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": 0.0, "mean": 0.0, "p5": 0.0, "p50": 0.0, "p95": 0.0}
    array = np.array(values, dtype=np.float32)
    return {
        "count": int(array.shape[0]),
        "min": float(array.min()),
        "mean": float(array.mean()),
        "p5": percentile(values, 5),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate UNKNOWN threshold.")
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--percentile", type=float, default=5.0)
    args = parser.parse_args()

    gallery_embeddings, _, gallery_meta = load_embeddings(args.cache_dir, "gallery")
    query_embeddings, _, query_meta = load_embeddings(args.cache_dir, "query")

    gallery_labels = gallery_meta.get("labels", [])
    query_labels = query_meta.get("labels", [])

    similarity = cosine_similarity(query_embeddings, gallery_embeddings)
    top1_idx = np.argmax(similarity, axis=1)
    top1_scores = similarity[np.arange(similarity.shape[0]), top1_idx]

    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    for i, idx in enumerate(top1_idx):
        if gallery_labels[idx] == query_labels[i]:
            correct_scores.append(float(top1_scores[i]))
        else:
            wrong_scores.append(float(top1_scores[i]))

    correct_stats = summarize(correct_scores)
    wrong_stats = summarize(wrong_scores)

    print("Top-1 similarity stats (correct):", correct_stats)
    print("Top-1 similarity stats (wrong):  ", wrong_stats)

    if correct_scores:
        suggested = percentile(correct_scores, args.percentile)
        print(
            f"Suggested threshold (p{args.percentile} of correct scores): {suggested:.4f}"
        )
    else:
        print("Suggested threshold: 0.0 (no correct matches found)")


if __name__ == "__main__":
    main()
