"""Frame index: build, persist, and cosine-rank."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .embeddings import Embedder, l2_normalize
from .frames import generate_dataset


@dataclass
class SearchHit:
    id: str
    title: str
    caption: str
    path: str
    score: float
    tags: list[str]


class FrameIndex:
    def __init__(self, root: Path, embedder: Embedder):
        self.root = Path(root)
        self.frames_dir = self.root / "frames"
        self.embedder = embedder
        self.catalog: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None  # (N, D)

    def ensure_dataset(self) -> None:
        catalog_path = self.root / "catalog.json"
        if not catalog_path.exists() or not self.frames_dir.exists():
            generate_dataset(self.frames_dir)
        self.catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    def build(self) -> None:
        self.ensure_dataset()
        vectors = []
        for item in self.catalog:
            img = Image.open(self.frames_dir / item["path"]).convert("RGB")
            # fuse image embedding with caption text for stronger text queries
            img_vec = self.embedder.embed_image(img)
            txt_vec = self.embedder.embed_text(item["caption"] + " " + " ".join(item.get("tags", [])))
            vectors.append(l2_normalize(0.65 * img_vec + 0.35 * txt_vec))
        self.matrix = np.stack(vectors, axis=0).astype(np.float32)
        self.save()

    def save(self) -> None:
        assert self.matrix is not None
        np.savez_compressed(self.root / "index.npz", matrix=self.matrix)
        meta = {"dim": int(self.matrix.shape[1]), "count": int(self.matrix.shape[0])}
        (self.root / "index.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def load(self) -> None:
        self.ensure_dataset()
        npz = self.root / "index.npz"
        if not npz.exists():
            self.build()
            return
        self.matrix = np.load(npz)["matrix"]

    def search_vector(self, query: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        if self.matrix is None:
            self.load()
        assert self.matrix is not None
        q = l2_normalize(query)
        scores = self.matrix @ q
        k = max(1, min(top_k, len(self.catalog)))
        idxs = np.argsort(-scores)[:k]
        hits: list[SearchHit] = []
        for i in idxs:
            item = self.catalog[int(i)]
            hits.append(
                SearchHit(
                    id=item["id"],
                    title=item["title"],
                    caption=item["caption"],
                    path=item["path"],
                    score=float(scores[int(i)]),
                    tags=list(item.get("tags", [])),
                )
            )
        return hits

    def search_text(self, text: str, top_k: int = 5) -> list[SearchHit]:
        return self.search_vector(self.embedder.embed_text(text), top_k=top_k)

    def search_image(self, image: Image.Image, top_k: int = 5) -> list[SearchHit]:
        return self.search_vector(self.embedder.embed_image(image), top_k=top_k)
