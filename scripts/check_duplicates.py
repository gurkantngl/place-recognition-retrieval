import argparse
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.manifest import filter_split, load_manifest, resolve_paths


def average_hash(image: Image.Image, hash_size: int = 8) -> np.ndarray:
    gray = image.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    pixels = np.array(gray, dtype=np.float32)
    return (pixels > pixels.mean()).astype(np.uint8).reshape(-1)


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    return int(np.count_nonzero(hash_a != hash_b))


def load_hashes(paths: list[str], hash_size: int, min_size: int) -> dict[str, np.ndarray]:
    hashes: dict[str, np.ndarray] = {}
    for path in paths:
        try:
            with Image.open(path) as img:
                if img.width < min_size or img.height < min_size:
                    continue
                hashes[path] = average_hash(img, hash_size=hash_size)
        except Exception:
            continue
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser(description="Check near-duplicates across splits.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument("--hash-size", type=int, default=8)
    parser.add_argument("--hamming-threshold", type=int, default=5)
    parser.add_argument("--min-size", type=int, default=64)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    gallery_rows = filter_split(rows, "gallery")
    query_rows = filter_split(rows, "query")

    gallery_paths = resolve_paths(gallery_rows, args.dataset_root)
    query_paths = resolve_paths(query_rows, args.dataset_root)

    gallery_hashes = load_hashes(gallery_paths, args.hash_size, args.min_size)
    query_hashes = load_hashes(query_paths, args.hash_size, args.min_size)

    collisions: list[tuple[str, str, int]] = []
    for q_path, q_hash in query_hashes.items():
        best = None
        best_dist = 10**9
        for g_path, g_hash in gallery_hashes.items():
            dist = hamming_distance(q_hash, g_hash)
            if dist < best_dist:
                best_dist = dist
                best = g_path
        if best is not None and best_dist <= args.hamming_threshold:
            collisions.append((q_path, best, best_dist))

    print(f"Gallery hashes: {len(gallery_hashes)} | Query hashes: {len(query_hashes)}")
    print(f"Potential near-duplicates (<= {args.hamming_threshold}): {len(collisions)}")
    for q_path, g_path, dist in collisions:
        print(f"query={q_path} gallery={g_path} hamming={dist}")


if __name__ == "__main__":
    main()
