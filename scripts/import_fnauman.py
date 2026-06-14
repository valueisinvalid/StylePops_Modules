#!/usr/bin/env python3
"""fnauman ikinci-el moda veri setinden FİLTRELİ takviye parçaları çeker.

Kaynak: fnauman/fashion-second-hand-front-only-rgb (CC-BY 4.0)
- Parquet shard'ları doğrudan indirilir (hf_hub) → rate limit / süre dolması yok
- Yalnızca dış giyim (kaban/mont/ceket) + sıcak ara katman alınır → gardırop
  kaban/çeşit açığını kapatır
- material (NIR tarayıcı) → fabric_composition'a parse edilir
- Çıktı: data/visual/garments_fnauman.json + data/assets/garments/fnauman/*.jpg

Kullanım: python scripts/import_fnauman.py --max-outer 800 --max-mid 600
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "data" / "visual"
IMG_DIR = ROOT / "data" / "assets" / "garments" / "fnauman"
OUT_PATH = VISUAL / "garments_fnauman.json"

DATASET = "fnauman/fashion-second-hand-front-only-rgb"
N_SHARDS = 13

# fnauman 'type' → (layer_role, subcategory, thermal_category, season_primary, bucket)
# bucket: kota grubu (heavy outer otomatik HEAVY_TYPES'tan belirlenir)
TYPE_MAP = {
    # Dış giyim
    "Winter Jacket": ("outer", "padded_coat", "kalin_mont", "kis", "outer"),
    "Outerwear": ("outer", "coat", "kaban", "kis", "outer"),
    "Coat": ("outer", "coat", "kaban", "kis", "outer"),
    "Rain Jacket": ("outer", "raincoat", "yagmurluk", "sonbahar", "outer"),
    "Jacket": ("outer", "jacket", "ince_ceket", "sonbahar", "outer"),
    "Jacker": ("outer", "jacket", "ince_ceket", "sonbahar", "outer"),
    "Denim Jacket": ("outer", "jacket", "ince_ceket", "sonbahar", "outer"),
    "Blazer": ("outer", "blazer", "ince_ceket", "sonbahar", "outer"),
    "Vest": ("outer", "jacket", "ince_ceket", "sonbahar", "outer"),
    # Sıcak ara katman
    "Sweater": ("mid", "sweater", "kalin_yun_kazak", "kis", "mid"),
    "Cardigan": ("mid", "cardigan", "ince_triko", "sonbahar", "mid"),
    "Hoodie": ("mid", "hoodie", "polar_sweatshirt", "sonbahar", "mid"),
    # Üstler
    "Top": ("base", "tshirt", "ince_pamuklu_tisort", "ilkbahar", "top"),
    "T-shirt": ("base", "tshirt", "ince_pamuklu_tisort", "ilkbahar", "top"),
    "Tank Top": ("base", "tank_top", "ince_pamuklu_tisort", "yaz", "top"),
    "Training Top": ("base", "tshirt", "ince_pamuklu_tisort", "ilkbahar", "top"),
    "Blouse": ("mid", "blouse", "pamuklu_gomlek", "ilkbahar", "top"),
    "Shirt": ("mid", "shirt", "pamuklu_gomlek", "sonbahar", "top"),
    "Tunic": ("mid", "blouse", "pamuklu_gomlek", "ilkbahar", "top"),
    # Altlar (özellikle düzgün pantolon — iki cinsiyet)
    "Trousers": ("bottom", "trousers", "chino_pantolon", "sonbahar", "bottom"),
    "Jeans": ("bottom", "jeans", "jean_pantolon", "sonbahar", "bottom"),
    "Winter Trousers": ("bottom", "trousers", "jean_pantolon", "kis", "bottom"),
    "Rain Trousers": ("bottom", "trousers", "chino_pantolon", "sonbahar", "bottom"),
    "Skirt": ("bottom", "skirt", "etek", "ilkbahar", "bottom"),
    "Shorts": ("bottom", "shorts", "sort", "yaz", "bottom"),
}

SEASON_USABLE = {
    "kalin_mont": ["kis"], "kaban": ["kis"], "trench_coat": ["sonbahar", "kis"],
    "yagmurluk": ["sonbahar", "ilkbahar"], "ince_ceket": ["sonbahar", "kis"],
    "kalin_yun_kazak": ["kis", "sonbahar"], "ince_triko": ["sonbahar", "ilkbahar"],
    "polar_sweatshirt": ["sonbahar", "kis"],
    "ince_pamuklu_tisort": ["ilkbahar", "yaz", "sonbahar"],
    "pamuklu_gomlek": ["ilkbahar", "sonbahar", "kis"],
    "chino_pantolon": ["ilkbahar", "sonbahar", "kis"],
    "jean_pantolon": ["sonbahar", "kis", "ilkbahar"],
    "etek": ["ilkbahar", "yaz"], "sort": ["yaz"],
}

GENDER_MAP = {
    "Ladies": "women", "Women": "women", "Ladys": "women",
    "Men": "men", "Unisex": "unisex",
}

_MATERIAL_RE = re.compile(r"(\d+)\s*%\s*([A-Za-zÅÄÖåäö/ ]+)")
_MATERIAL_NORM = {
    "polyester": "polyester", "cotton": "cotton", "bomull": "cotton",
    "wool": "wool", "ull": "wool", "viscose": "viscose", "viskos": "viscose",
    "elastane": "elastane", "elastan": "elastane", "polyamide": "polyamide",
    "nylon": "nylon", "acrylic": "acrylic", "akryl": "acrylic", "linen": "linen",
    "lin": "linen", "silk": "silk", "siden": "silk", "cashmere": "cashmere",
    "modal": "modal", "lyocell": "lyocell", "spandex": "elastane",
}


def parse_material(material: str) -> list[dict]:
    if not material:
        return []
    out = []
    for pct, name in _MATERIAL_RE.findall(material):
        key = name.strip().lower()
        norm = _MATERIAL_NORM.get(key, key.split("/")[0].strip())
        try:
            out.append({"material": norm, "pct": int(pct)})
        except ValueError:
            continue
    return out


def _srgb_to_lab(r: int, g: int, b: int) -> dict:
    def lin(c):
        c /= 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    R, G, B = lin(r), lin(g), lin(b)
    X = R * 0.4124 + G * 0.3576 + B * 0.1805
    Y = R * 0.2126 + G * 0.7152 + B * 0.0722
    Z = R * 0.0193 + G * 0.1192 + B * 0.9505
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883

    def f(t):
        return t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(X / Xn), f(Y / Yn), f(Z / Zn)
    return {
        "L": round(116 * fy - 16),
        "a": round(500 * (fx - fy)),
        "b": round(200 * (fy - fz)),
    }


def mean_lab_from_image(img) -> dict:
    from PIL import Image
    im = img.convert("RGBA").resize((64, 64))
    px = im.load()
    rs = gs = bs = n = 0
    for y in range(64):
        for x in range(64):
            r, g, b, a = px[x, y]
            if a < 30:
                continue
            if r > 245 and g > 245 and b > 245:
                continue
            rs += r; gs += g; bs += b; n += 1
    if n == 0:
        return {"L": 70, "a": 0, "b": 0}
    return _srgb_to_lab(rs // n, gs // n, bs // n)


def image_from_cell(cell) -> "object | None":
    """Parquet'teki HF Image hücresinden (struct {bytes,path}) PIL döndür."""
    from PIL import Image
    data = None
    if isinstance(cell, dict):
        data = cell.get("bytes")
    elif isinstance(cell, (bytes, bytearray)):
        data = cell
    if not data:
        return None
    return Image.open(BytesIO(data))


def passes_filter(row: dict) -> bool:
    cat = (row.get("category") or "").strip()
    if cat == "Children" or cat not in GENDER_MAP:
        return False
    if (row.get("usage") or "") not in ("Reuse", "Export"):
        return False
    try:
        if row.get("condition") is not None and int(row["condition"]) < 3:
            return False
    except (ValueError, TypeError):
        pass
    for dmg in ("stains", "holes", "smell", "damage"):
        if str(row.get(dmg) or "").strip().lower() == "major":
            return False
    return row.get("type") in TYPE_MAP


def main() -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    from collections import defaultdict
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-outer", type=int, default=800)
    ap.add_argument("--max-mid", type=int, default=600)
    ap.add_argument("--max-bottom-men", type=int, default=400)
    ap.add_argument("--max-bottom-women", type=int, default=550)
    ap.add_argument("--max-top-men", type=int, default=300)
    ap.add_argument("--max-top-women", type=int, default=450)
    args = ap.parse_args()

    HEAVY_TYPES = {"Winter Jacket", "Outerwear", "Coat"}
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    picked: list[dict] = []
    counts: dict[str, int] = defaultdict(int)
    seq = 0

    def cap_for(bucket: str, is_heavy: bool, gender: str) -> tuple[str, int]:
        gkey = "men" if gender == "men" else "women"
        if bucket == "outer":
            return ("heavy", 10**9) if is_heavy else ("light", args.max_outer)
        if bucket == "mid":
            return ("mid", args.max_mid)
        if bucket == "bottom":
            return (f"bottom_{gkey}",
                    args.max_bottom_men if gkey == "men" else args.max_bottom_women)
        return (f"top_{gkey}",
                args.max_top_men if gkey == "men" else args.max_top_women)

    cols = ["image", "brand", "usage", "condition", "type", "category",
            "trend", "colors", "cut", "pattern", "season", "material",
            "stains", "holes", "smell", "damage"]

    for shard in range(N_SHARDS):
        fname = f"data/train-{shard:05d}-of-{N_SHARDS:05d}.parquet"
        print(f"[shard {shard+1}/{N_SHARDS}] indiriliyor… {dict(counts)}")
        local = hf_hub_download(DATASET, fname, repo_type="dataset")
        table = pq.read_table(local, columns=cols)
        rows = table.to_pylist()
        del table
        for row in rows:
            if not passes_filter(row):
                continue
            type_name = row["type"]
            layer, subcat, tc, season_p, bucket = TYPE_MAP[type_name]
            is_heavy = type_name in HEAVY_TYPES
            gender = GENDER_MAP.get((row.get("category") or "").strip(), "women")
            ckey, cap = cap_for(bucket, is_heavy, gender)
            if counts[ckey] >= cap:
                continue

            fseason = (row.get("season") or "").strip()
            if fseason == "Winter":
                season_p = "kis"
            elif fseason == "Autumn":
                season_p = "sonbahar"
            elif fseason == "Summer" and bucket == "mid":
                continue  # yazlık sweater istemiyoruz

            seq += 1
            gid = f"FN{seq:04d}"
            try:
                img = image_from_cell(row["image"])
                if img is None:
                    raise ValueError("görsel boş")
                lab = mean_lab_from_image(img)
                rgb = img.convert("RGB")
                rgb.thumbnail((512, 512))
                img_path = IMG_DIR / f"{gid}.jpg"
                rgb.save(img_path, "JPEG", quality=88)
            except Exception as e:
                print(f"  {gid} görsel hata: {e}", file=sys.stderr)
                seq -= 1
                continue

            material = row.get("material") or ""
            colors = row.get("colors") or []
            brand = (row.get("brand") or "").strip()
            if brand.lower() in ("not applicable", "na", "n/a", "none", "unknown", ""):
                brand = ""
            name = f"{brand} {type_name}".strip() or type_name
            g = {
                "id": gid,
                "name": name[:80],
                "description": f"{type_name}. Material: {material}. Colors: {', '.join(colors)}.",
                "category": {"outer": "outerwear", "bottom": "bottom"}.get(layer, "top"),
                "subcategory": subcat,
                "layer_role": layer,
                "fabric_composition": parse_material(material),
                "layers": None,
                "sleeve": None,
                "coverage_ratio": 0.85 if layer == "outer" else 0.7,
                "color_lab": lab,
                "season_primary": season_p,
                "season_usable": SEASON_USABLE.get(tc, ["sonbahar", "kis"]),
                "thermal_category": tc,
                "image_path": str(img_path.relative_to(ROOT)),
                "image_source_url": f"hf://{DATASET}",
                "source": "fnauman_secondhand",
                "source_id": gid,
                "license": "CC-BY-4.0",
                "license_attribution": "Nauman, F. (2024) Clothing Dataset for Second-Hand Fashion, Zenodo, doi:10.5281/zenodo.13788681",
                "snapshot_date": "2026-06-13",
                "active": True,
                "training_only": False,
                "gender": GENDER_MAP[row["category"].strip()],
                "fn_meta": {
                    "type": type_name, "season": fseason, "pattern": row.get("pattern"),
                    "cut": row.get("cut"), "material": material, "colors": colors,
                },
            }
            picked.append(g)
            counts[ckey] += 1
        del rows
        # her shard sonrası ara kayıt (kesinti olursa kaybolmasın)
        OUT_PATH.write_text(json.dumps({
            "schema_version": "1.0", "source": DATASET, "license": "CC-BY-4.0",
            "count": len(picked), "garments": picked,
        }, ensure_ascii=False), encoding="utf-8")

    payload = {
        "schema_version": "1.0",
        "source": DATASET,
        "license": "CC-BY-4.0",
        "count": len(picked),
        "garments": picked,
    }
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nKaydedildi → {OUT_PATH} ({len(picked)} parça) · {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
