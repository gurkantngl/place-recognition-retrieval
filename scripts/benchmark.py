import argparse
import os
import sys
import time

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import ResNetEmbedder, load_embeddings
from src.index import cosine_similarity
from src.manifest import filter_split, load_manifest, resolve_paths


def time_embedding(paths: list[str], device: str, batch_size: int) -> tuple[float, int]:
    embedder = ResNetEmbedder(device=device, batch_size=batch_size)
    start = time.perf_counter()
    result = embedder.embed_paths(paths)
    elapsed = time.perf_counter() - start
    return elapsed, result.embeddings.shape[0]


def time_search(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> float:
    start = time.perf_counter()
    _ = cosine_similarity(query_embeddings, gallery_embeddings)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark embedding and search speed.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", action="store_true", help="Run a warmup pass")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    gallery_rows = filter_split(rows, "gallery")
    query_rows = filter_split(rows, "query")
    gallery_paths = resolve_paths(gallery_rows, args.dataset_root)
    query_paths = resolve_paths(query_rows, args.dataset_root)

    if args.warmup:
        _ = time_embedding(gallery_paths[: min(4, len(gallery_paths))], args.device, args.batch_size)

    gallery_time, gallery_count = time_embedding(gallery_paths, args.device, args.batch_size)
    query_time, query_count = time_embedding(query_paths, args.device, args.batch_size)

    gallery_embeddings, _, _ = load_embeddings(args.cache_dir, "gallery")
    query_embeddings, _, _ = load_embeddings(args.cache_dir, "query")
    search_time = time_search(query_embeddings, gallery_embeddings)

    print("Embedding benchmark")
    print(f"  Gallery images: {gallery_count} | time: {gallery_time:.2f}s | {gallery_count / max(gallery_time, 1e-9):.2f} img/s")
    print(f"  Query images:   {query_count} | time: {query_time:.2f}s | {query_count / max(query_time, 1e-9):.2f} img/s")
    print("Search benchmark")
    print(f"  Similarity time: {search_time:.4f}s | queries: {query_embeddings.shape[0]} | gallery: {gallery_embeddings.shape[0]}")


if __name__ == "__main__":
    main()
