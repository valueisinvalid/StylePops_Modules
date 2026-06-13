#!/usr/bin/env python3
"""Kombin parça görsellerinden önizleme kolajı üretir."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stylepops_core import garment_slot

CELL = 200
CELL_AB = 168
LABEL_H = 22
PADDING = 10
BG = (248, 248, 250)

SLOT_LABEL_TR = {
    "base": "İÇ",
    "mid": "ÜST",
    "outer": "DIŞ",
    "bottom": "ALT",
    "dress": "ELBİSE",
    "footwear": "AYAKKABI",
    "accessory": "AKS",
}


def _slot_label(garment: dict) -> str:
    return SLOT_LABEL_TR.get(garment_slot(garment), garment_slot(garment).upper()[:4])


def _gentle_crop(img: Image.Image, slot: str) -> Image.Image:
    """Yalnızca aşırı uzun model fotoğraflarında hafif odak (AB'de kullanılmaz)."""
    w, h = img.size
    if h < 60 or w < 60 or h / max(w, 1) < 1.35:
        return img
    if slot == "footwear":
        top = int(h * 0.45)
    elif slot == "bottom":
        top = int(h * 0.30)
    elif slot == "dress":
        top = int(h * 0.05)
        return img.crop((0, top, w, int(h * 0.95)))
    elif slot in ("base", "mid", "outer"):
        top = 0
        return img.crop((0, top, w, int(h * 0.78)))
    return img


def _load_thumb(garment: dict, size: int, *, fit: bool = True) -> Image.Image | None:
    rel = garment.get("image_path")
    if not rel:
        return None
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        if not fit:
            img = _gentle_crop(img, garment_slot(garment))
            img.thumbnail((size, size), Image.Resampling.LANCZOS)
            return img
        canvas = Image.new("RGB", (size, size), BG)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        ox = (size - img.width) // 2
        oy = (size - img.height) // 2
        canvas.paste(img, (ox, oy))
        return canvas
    except OSError:
        return None


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.rectangle((x, y, x + 56, y + LABEL_H - 2), fill=(40, 40, 48))
    draw.text((x + 4, y + 2), text[:8], fill=(240, 240, 245))


def build_combo_collage(
    piece_ids: list[str],
    garments: dict[str, dict],
    out_path: Path,
    title: str = "",
) -> bool:
    items: list[tuple[Image.Image, str]] = []
    for pid in piece_ids:
        g = garments.get(pid)
        if not g:
            continue
        t = _load_thumb(g, CELL, fit=True)
        if t:
            items.append((t, _slot_label(g)))
    if not items:
        return False

    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    cell_h = CELL + LABEL_H
    w = cols * CELL + (cols + 1) * PADDING
    h = rows * cell_h + (rows + 1) * PADDING + (24 if title else 0)
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((PADDING, 4), title[:80], fill=(40, 40, 45))

    y0 = 28 if title else PADDING
    for i, (thumb, label) in enumerate(items):
        row, col = divmod(i, cols)
        x = PADDING + col * (CELL + PADDING)
        y = y0 + row * (cell_h + PADDING)
        canvas.paste(thumb, (x, y))
        _draw_label(draw, x, y + CELL - LABEL_H, label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=92)
    return True


def build_ab_collage(
    combo_a: list[str],
    combo_b: list[str],
    garments: dict[str, dict],
    out_path: Path,
    label_a: str = "A",
    label_b: str = "B",
) -> bool:
    """A/B: her yan 2×2 ızgara, tam görsel (kırpma yok)."""
    cols, rows = 2, 2
    cell = CELL_AB
    cell_block = cell + LABEL_H
    half_w = cols * cell_block + (cols + 1) * PADDING
    half_h = rows * cell_block + (rows + 1) * PADDING + 20
    canvas = Image.new("RGB", (half_w * 2 + PADDING, half_h), BG)
    draw = ImageDraw.Draw(canvas)

    for side, piece_ids, side_label in ((0, combo_a, label_a), (1, combo_b, label_b)):
        x_off = side * (half_w + PADDING)
        draw.text((x_off + PADDING, 4), side_label, fill=(30, 30, 35))
        for i, pid in enumerate(piece_ids[:4]):
            g = garments.get(pid)
            if not g:
                continue
            t = _load_thumb(g, cell, fit=True)
            if not t:
                continue
            row, col = divmod(i, cols)
            x = x_off + PADDING + col * (cell_block + PADDING)
            y = 22 + row * (cell_block + PADDING)
            canvas.paste(t, (x, y))
            _draw_label(draw, x, y + cell - 2, _slot_label(g))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=92)
    return True


def main() -> None:
    from inventory_loader import load_production_garments

    garments = load_production_garments()
    if not garments:
        print("Envanter yok")
        return
    sample = list(garments.keys())[:4]
    out = ROOT / "data" / "assets" / "combos" / "_collage_test.jpg"
    build_combo_collage(sample, garments, out, title="test")
    print(f"Test kolaj → {out}")


if __name__ == "__main__":
    main()
