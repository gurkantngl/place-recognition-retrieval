import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class GalleryIndex:
    embeddings: np.ndarray
    labels: List[str]
    paths: List[str]


def save_index(index_dir: str, index: GalleryIndex) -> None:
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "gallery_embeddings.npy"), index.embeddings)
    meta = {"labels": index.labels, "paths": index.paths}
    with open(os.path.join(index_dir, "gallery_meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def load_index(index_dir: str) -> GalleryIndex:
    embeddings = np.load(os.path.join(index_dir, "gallery_embeddings.npy"))
    with open(os.path.join(index_dir, "gallery_meta.json"), "r", encoding="utf-8") as file:
        meta = json.load(file)
    return GalleryIndex(embeddings=embeddings, labels=meta["labels"], paths=meta["paths"])


def cosine_search(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray, top_k: int) -> np.ndarray:
    similarities = query_embeddings @ gallery_embeddings.T
    top_k = min(top_k, gallery_embeddings.shape[0])
    return np.argsort(-similarities, axis=1)[:, :top_k]


def cosine_similarity(query_embeddings: np.ndarray, gallery_embeddings: np.ndarray) -> np.ndarray:
    return query_embeddings @ gallery_embeddings.T
