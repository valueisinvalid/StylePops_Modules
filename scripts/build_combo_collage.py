#!/usr/bin/env python3
"""Kombin parça görsellerinden önizleme kolajı üretir."""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stylepops_core import garment_slot

CELL = 200
CELL_AB_W = 168
CELL_AB_H = 210
LABEL_H = 20
PADDING = 10
GAP = 6
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

_FONT: ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None
_FONT_TR = False


def _load_font(size: int = 13) -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, bool]:
    global _FONT, _FONT_TR
    if _FONT is not None:
        return _FONT, _FONT_TR
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                _FONT = ImageFont.truetype(path, size)
                _FONT_TR = True
                return _FONT, _FONT_TR
            except OSError:
                continue
    _FONT = ImageFont.load_default()
    _FONT_TR = False
    return _FONT, _FONT_TR


def _slot_label(garment: dict) -> str:
    slot = garment_slot(garment)
    label = SLOT_LABEL_TR.get(slot, slot.upper()[:4])
    _, tr_ok = _load_font()
    if not tr_ok:
        ascii_map = {
            "İÇ": "IC",
            "ÜST": "UST",
            "DIŞ": "DIS",
            "ALT": "ALT",
            "ELBİSE": "ELBISE",
            "AYAKKABI": "AYAK",
            "AKS": "AKS",
        }
        label = ascii_map.get(label, label)
    return label


def _load_thumb(
    garment: dict,
    width: int,
    height: int | None = None,
) -> Image.Image | None:
    rel = garment.get("image_path")
    if not rel:
        return None
    path = ROOT / rel
    if not path.exists():
        return None
    h = height if height is not None else width
    try:
        img = Image.open(path).convert("RGB")
        canvas = Image.new("RGB", (width, h), BG)
        img.thumbnail((width, h), Image.Resampling.LANCZOS)
        ox = (width - img.width) // 2
        oy = (h - img.height) // 2
        canvas.paste(img, (ox, oy))
        return canvas
    except OSError:
        return None


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str) -> None:
    font, _ = _load_font()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    box_w = min(max(tw + 12, 52), 120)
    draw.rectangle((x, y, x + box_w, y + LABEL_H), fill=(40, 40, 48))
    draw.text((x + 6, y + 3), text, font=font, fill=(240, 240, 245))


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
        t = _load_thumb(g, CELL, CELL)
        if t:
            items.append((t, _slot_label(g)))
    if not items:
        return False

    n = len(items)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    slot_h = CELL + LABEL_H + GAP
    w = cols * CELL + (cols + 1) * PADDING
    h = rows * slot_h + (rows + 1) * PADDING + (24 if title else 0)
    canvas = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(canvas)
    if title:
        font, _ = _load_font()
        draw.text((PADDING, 4), title[:80], fill=(40, 40, 45), font=font)

    y0 = 28 if title else PADDING
    for i, (thumb, label) in enumerate(items):
        row, col = divmod(i, cols)
        x = PADDING + col * (CELL + PADDING)
        y = y0 + row * (slot_h + PADDING)
        canvas.paste(thumb, (x, y))
        _draw_label(draw, x, y + CELL + 2, label)

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
    """A/B: her yan 2×2 ızgara; etiket görselin altında, kırpma yok."""
    cols, rows = 2, 2
    cw, ch = CELL_AB_W, CELL_AB_H
    slot_w = cw
    slot_h = ch + LABEL_H + GAP
    half_w = cols * slot_w + (cols + 1) * PADDING
    half_h = rows * slot_h + (rows + 1) * PADDING + 22
    canvas = Image.new("RGB", (half_w * 2 + PADDING, half_h), BG)
    draw = ImageDraw.Draw(canvas)
    font, _ = _load_font(15)

    for side, piece_ids, side_label in ((0, combo_a, label_a), (1, combo_b, label_b)):
        x_off = side * (half_w + PADDING)
        draw.text((x_off + PADDING, 4), side_label, fill=(30, 30, 35), font=font)
        for i, pid in enumerate(piece_ids[:4]):
            g = garments.get(pid)
            if not g:
                continue
            t = _load_thumb(g, cw, ch)
            if not t:
                continue
            row, col = divmod(i, cols)
            x = x_off + PADDING + col * (slot_w + PADDING)
            y = 24 + row * (slot_h + PADDING)
            canvas.paste(t, (x, y))
            _draw_label(draw, x, y + ch + 2, _slot_label(g))

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
