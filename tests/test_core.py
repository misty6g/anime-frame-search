from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.embeddings import HashEmbedder, l2_normalize
from app.frames import SCENES, generate_dataset, render_scene
from app.index import FrameIndex


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    generate_dataset(tmp_path / "frames")
    return tmp_path


@pytest.fixture()
def index(data_dir: Path) -> FrameIndex:
    idx = FrameIndex(data_dir, HashEmbedder(dim=128))
    idx.build()
    return idx


def test_l2_normalize_unit_length():
    v = l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
    assert pytest.approx(float(np.linalg.norm(v)), rel=1e-5) == 1.0


def test_hash_embedder_is_deterministic():
    e = HashEmbedder(64)
    a = e.embed_text("neon city night")
    b = e.embed_text("neon city night")
    assert np.allclose(a, b)


def test_hash_embedder_different_texts_differ():
    e = HashEmbedder(64)
    a = e.embed_text("snowy mountain shrine")
    b = e.embed_text("neon rainy city street")
    assert float(np.dot(a, b)) < 0.99


def test_render_scene_size_and_mode():
    img = render_scene(SCENES[0], size=(160, 90))
    assert img.size == (160, 90)
    assert img.mode == "RGB"


def test_generate_dataset_writes_catalog_and_pngs(data_dir: Path):
    catalog = data_dir / "catalog.json"
    assert catalog.exists()
    frames = list((data_dir / "frames").glob("*.png"))
    assert len(frames) == len(SCENES)
    assert len(frames) >= 8


def test_index_build_and_persist(data_dir: Path):
    idx = FrameIndex(data_dir, HashEmbedder(96))
    idx.build()
    assert (data_dir / "index.npz").exists()
    assert idx.matrix is not None
    assert idx.matrix.shape[0] == len(SCENES)


def test_text_search_ranks_neon_near_top(index: FrameIndex):
    hits = index.search_text("neon city night rain pink cyan", top_k=3)
    assert hits
    ids = [h.id for h in hits]
    assert "neon-city-night-02" in ids
    assert hits[0].score >= hits[-1].score


def test_text_search_cherry_blossom(index: FrameIndex):
    hits = index.search_text("pink cherry blossom train spring", top_k=3)
    assert any("cherry" in h.id for h in hits)


def test_image_search_self_retrieval(index: FrameIndex):
    path = index.frames_dir / index.catalog[0]["path"]
    img = Image.open(path).convert("RGB")
    hits = index.search_image(img, top_k=1)
    assert hits[0].id == index.catalog[0]["id"]
    assert hits[0].score > 0.5


def test_api_health_and_text_search(data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFS_DATA", str(data_dir))
    monkeypatch.setenv("AFS_EMBEDDER", "hash")
    import importlib

    import app.api as api

    importlib.reload(api)
    client = TestClient(api.app)
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    res = client.get("/api/search/text", params={"q": "space colony earth stars", "top_k": 3})
    assert res.status_code == 200
    body = res.json()
    assert body["results"]
    assert "space" in body["results"][0]["id"] or any("space" in r["id"] for r in body["results"])


def test_api_image_search(data_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AFS_DATA", str(data_dir))
    monkeypatch.setenv("AFS_EMBEDDER", "hash")
    import importlib

    import app.api as api

    importlib.reload(api)
    client = TestClient(api.app)
    png = next((data_dir / "frames").glob("*.png"))
    raw = png.read_bytes()
    res = client.post(
        "/api/search/image",
        files={"file": ("q.png", io.BytesIO(raw), "image/png")},
        params={"top_k": 2},
    )
    assert res.status_code == 200
    assert len(res.json()["results"]) == 2
