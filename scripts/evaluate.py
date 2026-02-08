import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import load_embeddings
from src.index import cosine_similarity
from src.metrics import mean_average_precision, recall_at_k


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval metrics.")
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--k-values", default="1,5,10")
    args = parser.parse_args()

    gallery_embeddings, _, gallery_meta = load_embeddings(args.cache_dir, "gallery")
    query_embeddings, _, query_meta = load_embeddings(args.cache_dir, "query")

    gallery_labels = gallery_meta.get("labels", [])
    query_labels = query_meta.get("labels", [])

    similarity = cosine_similarity(query_embeddings, gallery_embeddings)

    k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    for k in k_values:
        recall = recall_at_k(similarity, query_labels, gallery_labels, k)
        print(f"Recall@{k}: {recall:.4f}")

    map_score = mean_average_precision(similarity, query_labels, gallery_labels)
    print(f"mAP: {map_score:.4f}")


if __name__ == "__main__":
    main()
