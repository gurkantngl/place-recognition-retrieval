import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import models

from src.image_io import read_image_rgb


@dataclass
class EmbeddingResult:
    embeddings: np.ndarray
    paths: List[str]
    skipped: List[Tuple[str, str]]


class ResNetEmbedder:
    def __init__(self, device: str = "cpu", batch_size: int = 32):
        self.device = torch.device(device)
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        self.feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.feature_extractor.eval().to(self.device)
        self.transform = weights.transforms()
        self.batch_size = batch_size

    def _preprocess(self, image_rgb: np.ndarray) -> torch.Tensor:
        pil_image = Image.fromarray(image_rgb)
        return self.transform(pil_image)

    @torch.no_grad()
    def embed_paths(self, image_paths: List[str]) -> EmbeddingResult:
        tensors: List[torch.Tensor] = []
        kept_paths: List[str] = []
        skipped: List[Tuple[str, str]] = []
        embeddings: List[np.ndarray] = []

        def flush_batch() -> None:
            if not tensors:
                return
            batch = torch.stack(tensors).to(self.device)
            feats = self.feature_extractor(batch).squeeze(-1).squeeze(-1)
            feats = torch.nn.functional.normalize(feats, p=2, dim=1)
            embeddings.append(feats.cpu().numpy())
            tensors.clear()

        for path in image_paths:
            image_rgb, status = read_image_rgb(path)
            if image_rgb is None:
                skipped.append((path, status))
                continue
            try:
                tensor = self._preprocess(image_rgb)
            except Exception:
                skipped.append((path, "preprocess_error"))
                continue
            tensors.append(tensor)
            kept_paths.append(path)
            if len(tensors) >= self.batch_size:
                flush_batch()
        flush_batch()
        if embeddings:
            array = np.concatenate(embeddings, axis=0)
        else:
            array = np.zeros((0, 2048), dtype=np.float32)
        return EmbeddingResult(embeddings=array, paths=kept_paths, skipped=skipped)


def save_embeddings(
    cache_dir: str,
    split: str,
    embeddings: np.ndarray,
    paths: List[str],
    labels: List[str],
    cache_key: str,
    skipped: List[Tuple[str, str]],
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(cache_dir, split)
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "embeddings.npy"), embeddings)
    meta = {
        "split": split,
        "cache_key": cache_key,
        "paths": paths,
        "labels": labels,
        "skipped": skipped,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as file:
        json.dump(meta, file, ensure_ascii=False, indent=2)


def load_embeddings(cache_dir: str, split: str) -> Tuple[np.ndarray, List[str], dict]:
    out_dir = os.path.join(cache_dir, split)
    embeddings = np.load(os.path.join(out_dir, "embeddings.npy"))
    with open(os.path.join(out_dir, "meta.json"), "r", encoding="utf-8") as file:
        meta = json.load(file)
    return embeddings, meta["paths"], meta
