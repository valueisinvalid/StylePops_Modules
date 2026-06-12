#!/usr/bin/env python3
"""Kombin parça görsellerinden önizleme kolajı üretir."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
CELL = 200
PADDING = 8
BG = (248, 248, 250)


from inventory_loader import load_production_garments as load_garments


def _load_thumb(garment: dict) -> Image.Image | None:
    rel = garment.get("image_path")
    if not rel:
        return None
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((CELL, CELL), Image.Resampling.LANCZOS)
        return img
    except OSError:
        return None


def build_combo_collage(
    piece_ids: list[str],
    garments: dict[str, dict],
    out_path: Path,
    title: str = "",
) -> bool:
    thumbs = []
    for pid in piece_ids:
        g = garments.get(pid)
        if not g:
            continue
        t = _load_thumb(g)
        if t:
            thumbs.append(t)
    if not thumbs:
        return False

    n = len(thumbs)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    w = cols * CELL + (cols + 1) * PADDING
    h = rows * CELL + (rows + 1) * PADDING + (24 if title else 0)
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((PADDING, 4), title[:80], fill=(40, 40, 45))

    y0 = 28 if title else PADDING
    for i, thumb in enumerate(thumbs):
        row, col = divmod(i, cols)
        x = PADDING + col * (CELL + PADDING)
        y = y0 + row * (CELL + PADDING)
        ox = x + (CELL - thumb.width) // 2
        oy = y + (CELL - thumb.height) // 2
        canvas.paste(thumb, (ox, oy))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=88)
    return True


def build_ab_collage(
    combo_a: list[str],
    combo_b: list[str],
    garments: dict[str, dict],
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
) -> bool:
    half_w = 2 * CELL + 3 * PADDING
    h = CELL + 2 * PADDING + 20
    canvas = Image.new("RGB", (half_w * 2 + PADDING, h), BG)
    draw = ImageDraw.Draw(canvas)

    for side, piece_ids, label in ((0, combo_a, label_a), (1, combo_b, label_b)):
        x_off = side * (half_w + PADDING)
        draw.text((x_off + PADDING, 4), label, fill=(30, 30, 35))
        for i, pid in enumerate(piece_ids[:4]):
            g = garments.get(pid)
            if not g:
                continue
            t = _load_thumb(g)
            if not t:
                continue
            x = x_off + PADDING + i * (CELL // 2 + 4)
            y = 22
            small = t.copy()
            small.thumbnail((CELL // 2, CELL // 2), Image.Resampling.LANCZOS)
            canvas.paste(small, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=88)
    return True
