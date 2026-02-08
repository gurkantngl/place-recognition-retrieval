import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import ResNetEmbedder, load_embeddings, save_embeddings
from src.manifest import filter_split, load_manifest, resolve_paths, split_hash


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract embeddings for a split.")
    parser.add_argument("--manifest", required=True, help="Path to manifest.csv")
    parser.add_argument("--dataset-root", required=True, help="Dataset root directory")
    parser.add_argument("--split", required=True, choices=["gallery", "query"])
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    split_rows = filter_split(rows, args.split)
    cache_key = split_hash(rows, args.split)
    cache_meta_path = os.path.join(args.cache_dir, args.split, "meta.json")

    if os.path.exists(cache_meta_path):
        with open(cache_meta_path, "r", encoding="utf-8") as file:
            meta = json.load(file)
        if meta.get("cache_key") == cache_key:
            embeddings, paths, meta = load_embeddings(args.cache_dir, args.split)
            print(f"Cache hit: {args.split} -> {embeddings.shape}")
            print(f"Kept: {len(paths)} | Skipped: {len(meta.get('skipped', []))}")
            return

    paths = resolve_paths(split_rows, args.dataset_root)
    labels = [row.label for row in split_rows]

    embedder = ResNetEmbedder(device=args.device, batch_size=args.batch_size)
    result = embedder.embed_paths(paths)

    kept_labels = []
    kept_paths = []
    label_by_path = dict(zip(paths, labels))
    for path in result.paths:
        kept_paths.append(path)
        kept_labels.append(label_by_path[path])

    save_embeddings(
        cache_dir=args.cache_dir,
        split=args.split,
        embeddings=result.embeddings,
        paths=kept_paths,
        labels=kept_labels,
        cache_key=cache_key,
        skipped=result.skipped,
    )

    print(f"Saved {args.split} embeddings -> {result.embeddings.shape}")
    print(f"Kept: {len(kept_paths)} | Skipped: {len(result.skipped)}")


if __name__ == "__main__":
    main()
