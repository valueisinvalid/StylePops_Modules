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
CELL_AB = 200
LABEL_H = 20
PADDING = 8
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


def _smart_crop(img: Image.Image, slot: str) -> Image.Image:
    """Model fotoğraflarında slot'a göre ilgili bölgeyi göster (alt üst karışmasın)."""
    w, h = img.size
    if h < 40 or w < 40:
        return img
    if slot == "bottom":
        box = (0, int(h * 0.38), w, h)
    elif slot == "footwear":
        box = (0, int(h * 0.52), w, h)
    elif slot in ("base", "mid", "outer"):
        box = (0, 0, w, int(h * 0.62))
    elif slot == "dress":
        box = (0, int(h * 0.08), w, int(h * 0.92))
    else:
        box = (int(w * 0.1), int(h * 0.15), int(w * 0.9), int(h * 0.85))
    return img.crop(box)


def _load_thumb(garment: dict, size: int) -> Image.Image | None:
    rel = garment.get("image_path")
    if not rel:
        return None
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        img = Image.open(path).convert("RGB")
        slot = garment_slot(garment)
        img = _smart_crop(img, slot)
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        return img
    except OSError:
        return None


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    draw.rectangle((x, y, x + 52, y + LABEL_H - 2), fill=(40, 40, 48))
    draw.text((x + 3, y + 1), text[:8], fill=(240, 240, 245))


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
        t = _load_thumb(g, CELL)
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
        ox = x + (CELL - thumb.width) // 2
        oy = y + (CELL - thumb.height) // 2
        canvas.paste(thumb, (ox, oy))
        _draw_label(draw, x, y + CELL - LABEL_H, label)

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
    max_pieces = 4
    thumb_size = CELL_AB
    cell_w = thumb_size + 6
    half_w = max_pieces * cell_w + 2 * PADDING
    row_h = thumb_size + LABEL_H + 24
    canvas = Image.new("RGB", (half_w * 2 + PADDING, row_h), BG)
    draw = ImageDraw.Draw(canvas)

    for side, piece_ids, side_label in ((0, combo_a, label_a), (1, combo_b, label_b)):
        x_off = side * (half_w + PADDING)
        draw.text((x_off + PADDING, 4), side_label, fill=(30, 30, 35))
        for i, pid in enumerate(piece_ids[:max_pieces]):
            g = garments.get(pid)
            if not g:
                continue
            t = _load_thumb(g, thumb_size)
            if not t:
                continue
            x = x_off + PADDING + i * cell_w
            y = 22
            ox = x + (thumb_size - t.width) // 2
            oy = y + (thumb_size - t.height) // 2
            canvas.paste(t, (ox, oy))
            _draw_label(draw, x, y + thumb_size - LABEL_H, _slot_label(g))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format="JPEG", quality=88)
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
