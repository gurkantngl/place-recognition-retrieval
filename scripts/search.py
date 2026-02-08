import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import ResNetEmbedder
from src.index import load_index


def collect_query_paths(query_path: str) -> list[str]:
    if os.path.isdir(query_path):
        paths = []
        for name in sorted(os.listdir(query_path)):
            if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                paths.append(os.path.join(query_path, name))
        return paths
    return [query_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Top-K matches.")
    parser.add_argument("--query", required=True, help="Query image or folder")
    parser.add_argument("--index-dir", default="cache/index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    args = parser.parse_args()

    index = load_index(args.index_dir)
    embedder = ResNetEmbedder(device=args.device, batch_size=args.batch_size)
    query_paths = collect_query_paths(args.query)

    result = embedder.embed_paths(query_paths)
    if result.embeddings.shape[0] == 0:
        print("No valid query images.")
        return

    similarity = result.embeddings @ index.embeddings.T
    for i, path in enumerate(result.paths):
        scores = similarity[i]
        top_k = min(args.top_k, scores.shape[0])
        order = np.argsort(-scores)[:top_k]
        top1_score = float(scores[order[0]]) if order.size > 0 else 0.0
        top2_score = float(scores[order[1]]) if order.size > 1 else 0.0

        is_unknown = False
        if args.threshold is not None and top1_score < args.threshold:
            is_unknown = True
        if args.margin is not None and (top1_score - top2_score) < args.margin:
            is_unknown = True

        print(f"\nQuery: {path}")
        if is_unknown:
            print("Prediction: UNKNOWN")
        for rank, idx in enumerate(order, start=1):
            print(
                f"  {rank}. label={index.labels[idx]} score={scores[idx]:.4f} path={index.paths[idx]}"
            )

    if result.skipped:
        print(f"\nSkipped {len(result.skipped)} query images.")


if __name__ == "__main__":
    main()
