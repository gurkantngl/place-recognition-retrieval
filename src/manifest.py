import csv
import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Dict


@dataclass(frozen=True)
class ManifestRow:
    mapping_line: str
    class_name: str
    class_dir: str
    landmark_id: str
    split: str
    image_id: str
    relpath: str
    url: str

    @property
    def label(self) -> str:
        return self.landmark_id


def load_manifest(manifest_path: str) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for raw in reader:
            rows.append(
                ManifestRow(
                    mapping_line=raw.get("mapping_line", ""),
                    class_name=raw.get("class_name", ""),
                    class_dir=raw.get("class_dir", ""),
                    landmark_id=raw.get("landmark_id", ""),
                    split=raw.get("split", ""),
                    image_id=raw.get("image_id", ""),
                    relpath=raw.get("relpath", ""),
                    url=raw.get("url", ""),
                )
            )
    return rows


def filter_split(rows: Iterable[ManifestRow], split: str) -> List[ManifestRow]:
    return [row for row in rows if row.split == split]


def resolve_paths(rows: Iterable[ManifestRow], dataset_root: str) -> List[str]:
    return [os.path.join(dataset_root, row.relpath) for row in rows]


def split_hash(rows: Iterable[ManifestRow], split: str) -> str:
    hasher = hashlib.sha1()
    items = sorted(row.relpath for row in rows if row.split == split)
    for relpath in items:
        hasher.update(relpath.encode("utf-8"))
        hasher.update(b"\n")
    hasher.update(split.encode("utf-8"))
    return hasher.hexdigest()
