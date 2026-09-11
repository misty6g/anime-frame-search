"""FastAPI service for text/image frame search."""

from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from .embeddings import get_embedder
from .index import FrameIndex

ROOT = Path(os.environ.get("AFS_DATA", Path(__file__).resolve().parent.parent / "data"))
BACKEND = os.environ.get("AFS_EMBEDDER", "hash")  # default hash for fast boot; set clip for full CLIP

app = FastAPI(title="Anime Frame Search", version="1.0.0")
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

_embedder = get_embedder(BACKEND)
_index = FrameIndex(ROOT, _embedder)
_index.load()


class TextQuery(BaseModel):
    query: str
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    page = static_dir / "index.html"
    if page.exists():
        return page.read_text(encoding="utf-8")
    return "<h1>Anime Frame Search</h1><p>API up. Add static/index.html for UI.</p>"


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "frames": len(_index.catalog),
        "embedder": type(_embedder).__name__,
        "dim": _embedder.dim,
    }


@app.post("/api/search/text")
def search_text(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(400, "query required")
    hits = _index.search_text(body.query, top_k=body.top_k)
    return {"query": body.query, "results": [h.__dict__ for h in hits]}


@app.get("/api/search/text")
def search_text_get(q: str = Query(...), top_k: int = 5):
    hits = _index.search_text(q, top_k=top_k)
    return {"query": q, "results": [h.__dict__ for h in hits]}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...), top_k: int = 5):
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"invalid image: {exc}") from exc
    hits = _index.search_image(img, top_k=top_k)
    return {"filename": file.filename, "results": [h.__dict__ for h in hits]}


@app.get("/frames/{name}")
def frame_file(name: str):
    path = _index.frames_dir / name
    if not path.exists():
        raise HTTPException(404, "frame not found")
    return FileResponse(path)
