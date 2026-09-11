# Anime Frame Search

Multimodal **visual search** over synthetic anime-style short-clip frames — text or image query → ranked frames.

Built for TikTok-style **AI Search / Visual Search** portfolio storytelling: embedding index, cosine retrieval, FastAPI serving, and a tiny browser UI.

> **Dataset policy:** frames are **original synthetic geometric art** (CC0-style / authored here). No copyrighted anime scrapes.

## Architecture

```
Query (text | image)
        │
        ▼
  Embedder backend
  ├─ ClipEmbedder (sentence-transformers CLIP ViT-B/32)
  └─ HashEmbedder (deterministic CI/demo fallback)
        │
        ▼
  FrameIndex (fused image+caption vectors, cosine rank)
        │
        ▼
  FastAPI  /  CLI  /  static UI
```

1. `generate` draws synthetic scenes → `data/frames/*.png` + `catalog.json`
2. `build` embeds each frame (image ⊕ caption/tags) → `index.npz`
3. Search embeds the query and returns top-k cosine hits

## Stack

- Python 3.11+
- **sentence-transformers** CLIP (`clip-ViT-B-32`) for production-quality embeddings
- **FastAPI** + **Uvicorn**
- **Pillow** / **NumPy**
- **pytest**

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # or: pip install -e ".[dev]" with a lighter set for hash-only

# Fast path (no model download) — HashEmbedder
export AFS_EMBEDDER=hash
python -m app.cli --embedder hash generate
python -m app.cli --embedder hash build
python -m app.cli --embedder hash search "neon city night rain"
python -m app.cli --embedder hash serve --port 8088
# open http://127.0.0.1:8088
```

### Full CLIP backend

```bash
export AFS_EMBEDDER=clip
python -m app.cli --embedder clip build
python -m app.cli --embedder clip serve --port 8088
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Liveness + frame count |
| GET/POST | `/api/search/text` | Text → ranked frames |
| POST | `/api/search/image` | Image upload → ranked frames |
| GET | `/frames/{name}` | Raw frame PNG |

## Tests

```bash
pip install pytest httpx fastapi pillow numpy pydantic python-multipart
AFS_EMBEDDER=hash pytest -q
```

≥8 tests cover embedding determinism, dataset generation, index persistence, text/image ranking, and API routes.

## License

MIT © Gyan Mistry
