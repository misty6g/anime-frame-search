"""Generate synthetic CC0 anime-style short-clip frames (no copyrighted scrapes)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

# Hand-authored synthetic scenes — geometric art only (CC0 / original).
SCENES: list[dict[str, Any]] = [
    {
        "id": "sunset-harbor-01",
        "title": "Sunset Harbor",
        "caption": "orange sunset over a quiet harbor with small boats and warm sky",
        "tags": ["sunset", "harbor", "boats", "orange", "sky"],
        "bg": (255, 140, 60),
        "accents": [(40, 60, 120), (255, 220, 120)],
    },
    {
        "id": "neon-city-night-02",
        "title": "Neon City Night",
        "caption": "rainy neon city street at night with pink and cyan glow",
        "tags": ["neon", "city", "night", "rain", "pink", "cyan"],
        "bg": (20, 18, 40),
        "accents": [(255, 40, 160), (40, 220, 255)],
    },
    {
        "id": "forest-spirit-03",
        "title": "Forest Spirit Path",
        "caption": "misty green forest path with soft light and glowing fireflies",
        "tags": ["forest", "green", "mist", "fireflies", "path"],
        "bg": (34, 90, 50),
        "accents": [(180, 255, 140), (255, 255, 160)],
    },
    {
        "id": "school-rooftop-04",
        "title": "School Rooftop",
        "caption": "sunny school rooftop with blue sky and a red fence railing",
        "tags": ["school", "rooftop", "blue", "sky", "fence"],
        "bg": (120, 190, 255),
        "accents": [(220, 50, 50), (250, 250, 250)],
    },
    {
        "id": "mecha-hangar-05",
        "title": "Mecha Hangar",
        "caption": "giant blue mecha standing in a metal hangar with sparks",
        "tags": ["mecha", "hangar", "blue", "metal", "robot"],
        "bg": (60, 70, 85),
        "accents": [(60, 120, 220), (255, 200, 80)],
    },
    {
        "id": "cherry-train-06",
        "title": "Cherry Blossom Train",
        "caption": "pink cherry blossom train platform in spring morning light",
        "tags": ["cherry", "blossom", "pink", "train", "spring"],
        "bg": (255, 200, 220),
        "accents": [(200, 80, 120), (90, 90, 100)],
    },
    {
        "id": "space-colony-07",
        "title": "Space Colony Window",
        "caption": "space colony window view of earth and stars from orbit",
        "tags": ["space", "colony", "earth", "stars", "orbit"],
        "bg": (8, 10, 30),
        "accents": [(80, 140, 255), (200, 220, 255)],
    },
    {
        "id": "beach-festival-08",
        "title": "Beach Festival Lanterns",
        "caption": "night beach festival with glowing lanterns and ocean waves",
        "tags": ["beach", "festival", "lanterns", "night", "ocean"],
        "bg": (15, 25, 55),
        "accents": [(255, 180, 60), (40, 100, 180)],
    },
    {
        "id": "snow-shrine-09",
        "title": "Snow Shrine",
        "caption": "snowy mountain shrine with red torii and quiet winter light",
        "tags": ["snow", "shrine", "winter", "red", "mountain"],
        "bg": (230, 240, 255),
        "accents": [(200, 40, 40), (180, 200, 220)],
    },
    {
        "id": "cyber-market-10",
        "title": "Cyber Market Alley",
        "caption": "crowded cyberpunk market alley with holographic signs",
        "tags": ["cyberpunk", "market", "alley", "hologram", "crowd"],
        "bg": (35, 20, 45),
        "accents": [(120, 255, 180), (255, 80, 200)],
    },
    {
        "id": "dragon-sky-11",
        "title": "Dragon Over Clouds",
        "caption": "fantasy dragon flying above bright clouds in golden hour",
        "tags": ["dragon", "clouds", "fantasy", "gold", "sky"],
        "bg": (255, 190, 100),
        "accents": [(90, 40, 140), (255, 255, 240)],
    },
    {
        "id": "rainy-cafe-12",
        "title": "Rainy Cafe Window",
        "caption": "cozy cafe window on a rainy afternoon with warm lamp light",
        "tags": ["cafe", "rain", "window", "cozy", "lamp"],
        "bg": (90, 100, 120),
        "accents": [(255, 210, 140), (60, 50, 40)],
    },
]


def render_scene(scene: dict[str, Any], size: tuple[int, int] = (384, 216)) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, scene["bg"])
    draw = ImageDraw.Draw(img)
    a0, a1 = scene["accents"]
    # stylized horizon / blocks — original geometric artwork
    draw.rectangle([0, int(h * 0.55), w, h], fill=a0)
    draw.ellipse([int(w * 0.65), int(h * 0.08), int(w * 0.92), int(h * 0.42)], fill=a1)
    draw.polygon(
        [(int(w * 0.1), h), (int(w * 0.28), int(h * 0.4)), (int(w * 0.46), h)],
        fill=a1,
    )
    draw.rectangle([int(w * 0.15), int(h * 0.35), int(w * 0.22), h], fill=a0)
    # frame id watermark (tiny)
    draw.text((8, 8), scene["id"], fill=(255, 255, 255))
    return img


def generate_dataset(out_dir: Path) -> list[dict[str, Any]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog: list[dict[str, Any]] = []
    for scene in SCENES:
        path = out_dir / f"{scene['id']}.png"
        render_scene(scene).save(path)
        entry = {
            "id": scene["id"],
            "title": scene["title"],
            "caption": scene["caption"],
            "tags": scene["tags"],
            "path": str(path.name),
        }
        catalog.append(entry)
    (out_dir.parent / "catalog.json").write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    return catalog
