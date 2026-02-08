import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.embedder import load_embeddings
from src.index import GalleryIndex, save_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build gallery index.")
    parser.add_argument("--cache-dir", default="cache/embeddings")
    parser.add_argument("--index-dir", default="cache/index")
    args = parser.parse_args()

    embeddings, paths, meta = load_embeddings(args.cache_dir, "gallery")
    labels = meta.get("labels", [])
    index = GalleryIndex(embeddings=embeddings, labels=labels, paths=paths)
    save_index(args.index_dir, index)
    print(f"Saved index -> {args.index_dir} | {embeddings.shape}")


if __name__ == "__main__":
    main()
