"""CLI: build index and run text searches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import get_embedder
from .frames import generate_dataset
from .index import FrameIndex


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="anime-frame-search", description="CLIP-style anime frame search")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--embedder", default="hash", choices=["hash", "clip", "auto"])
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("generate", help="Generate synthetic CC0 frames + catalog")
    sub.add_parser("build", help="Build embedding index")

    p_search = sub.add_parser("search", help="Text search against the index")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)

    p_serve = sub.add_parser("serve", help="Run FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8088)

    args = parser.parse_args(argv)
    embedder = get_embedder(args.embedder)
    index = FrameIndex(args.data, embedder)

    if args.cmd == "generate":
        catalog = generate_dataset(args.data / "frames")
        print(f"Generated {len(catalog)} frames → {args.data / 'frames'}")
        return

    if args.cmd == "build":
        index.build()
        print(f"Indexed {len(index.catalog)} frames (dim={embedder.dim})")
        return

    if args.cmd == "search":
        index.load()
        hits = index.search_text(args.query, top_k=args.top_k)
        print(json.dumps([h.__dict__ for h in hits], indent=2))
        return

    if args.cmd == "serve":
        import os

        import uvicorn

        os.environ["AFS_DATA"] = str(args.data.resolve())
        os.environ["AFS_EMBEDDER"] = args.embedder
        index.load()
        uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
