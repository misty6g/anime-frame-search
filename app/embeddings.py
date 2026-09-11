"""Pluggable embedding backends: deterministic demo + sentence-transformers CLIP."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from PIL import Image


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(vec))
    if n < 1e-12:
        return vec
    return vec / n


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_image(self, image: Image.Image) -> np.ndarray: ...

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self.embed_text(t) for t in texts], axis=0)


class HashEmbedder(Embedder):
    """Fast, deterministic embedder for tests/CI (no model download).

    Maps text tokens and coarse image color/layout stats into a shared vector
    space so text↔image ranking remains meaningful on synthetic demos.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _hash_buckets(self, parts: Iterable[str]) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for part in parts:
            digest = hashlib.sha256(part.encode("utf-8")).digest()
            for i in range(0, len(digest), 4):
                idx = int.from_bytes(digest[i : i + 2], "little") % self.dim
                sign = 1.0 if digest[i + 2] % 2 == 0 else -1.0
                mag = (digest[i + 3] + 1) / 256.0
                vec[idx] += sign * mag
        return l2_normalize(vec)

    def embed_text(self, text: str) -> np.ndarray:
        tokens = [t.lower() for t in text.replace("-", " ").split() if t]
        if not tokens:
            tokens = ["empty"]
        # also include bigrams for phrase sensitivity
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return self._hash_buckets(grams)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        # quadrant mean RGB + global histogram tags as pseudo-tokens
        parts: list[str] = []
        h, w, _ = arr.shape
        for y0, y1, ytag in ((0, h // 2, "top"), (h // 2, h, "bot")):
            for x0, x1, xtag in ((0, w // 2, "left"), (w // 2, w, "right")):
                patch = arr[y0:y1, x0:x1]
                r, g, b = patch.mean(axis=(0, 1))
                parts.append(f"{ytag}_{xtag}_r{int(r*8)}_g{int(g*8)}_b{int(b*8)}")
        # dominant channel cue
        mean = arr.mean(axis=(0, 1))
        dom = ("red", "green", "blue")[int(np.argmax(mean))]
        parts.append(f"dom_{dom}")
        bright = "bright" if mean.mean() > 0.55 else "dark"
        parts.append(bright)
        return self._hash_buckets(parts)


class ClipEmbedder(Embedder):
    """CLIP via sentence-transformers (`clip-ViT-B-32`)."""

    def __init__(self, model_name: str = "clip-ViT-B-32"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        # probe dim
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def embed_text(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        vec = self.model.encode([image.convert("RGB")], convert_to_numpy=True, normalize_embeddings=True)[0]
        return np.asarray(vec, dtype=np.float32)


def get_embedder(backend: str = "auto") -> Embedder:
    backend = (backend or "auto").lower()
    if backend == "hash":
        return HashEmbedder()
    if backend == "clip":
        return ClipEmbedder()
    if backend == "auto":
        try:
            return ClipEmbedder()
        except Exception:
            return HashEmbedder()
    raise ValueError(f"Unknown embedder backend: {backend}")
