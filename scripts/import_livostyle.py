#!/usr/bin/env python3
"""Livostyle MIT katalogundan görsel envanter importu — snapshot + mirror."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOKUPS = ROOT / "data" / "lookups"
VISUAL = ROOT / "data" / "visual"
ASSETS = ROOT / "data" / "assets" / "garments"

PRODUCTS_URL = (
    "https://raw.githubusercontent.com/arturayupov/"
    "womens-fashion-catalog-open-data/main/data/products.json"
)
THUMB_BASE = (
    "https://raw.githubusercontent.com/arturayupov/"
    "womens-fashion-catalog-open-data/main/thumbnails"
)

COLOR_HINTS: dict[str, tuple[float, float, float]] = {
    "black": (12, 0, 0),
    "white": (92, 0, 2),
    "cream": (88, 2, 12),
    "beige": (78, 4, 18),
    "brown": (38, 12, 22),
    "navy": (22, 8, -28),
    "blue": (45, 5, -35),
    "red": (42, 48, 28),
    "pink": (72, 28, 8),
    "green": (48, -22, 12),
    "olive": (42, -8, 22),
    "gray": (55, 0, 0),
    "grey": (55, 0, 0),
    "yellow": (82, -4, 52),
    "orange": (62, 32, 48),
    "purple": (38, 32, -18),
    "floral": (70, 18, 12),
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def fetch_json(url: str) -> list | dict:
    req = urllib.request.Request(url, headers={"User-Agent": "StylePops/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_file(url: str, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "StylePops/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def infer_color_lab(title: str, tags: list[str]) -> dict[str, float]:
    text = f"{title} {' '.join(tags)}".lower()
    for hint, lab in COLOR_HINTS.items():
        if hint in text:
            return {"L": lab[0], "a": lab[1], "b": lab[2]}
    return {"L": 60.0, "a": 5.0, "b": 8.0}


def parse_fabric(product: dict) -> list[dict]:
    meta = product.get("metafields") or {}
    fabric = (meta.get("fabric") or "").lower()
    if not fabric:
        return [{"material": "cotton", "pct": 100}]
    parts = []
    for token in re.split(r"[,+/]", fabric):
        token = token.strip()
        if not token:
            continue
        pct_match = re.search(r"(\d+)\s*%", token)
        pct = float(pct_match.group(1)) if pct_match else None
        for mat in ("cotton", "polyester", "wool", "linen", "viscose", "denim", "nylon", "silk", "elastane"):
            if mat in token:
                parts.append({"material": mat, "pct": pct or 0})
                break
    if not parts:
        if "wool" in fabric:
            parts = [{"material": "wool", "pct": 100}]
        elif "linen" in fabric:
            parts = [{"material": "linen", "pct": 100}]
        elif "denim" in fabric:
            parts = [{"material": "denim", "pct": 98}, {"material": "elastane", "pct": 2}]
        else:
            parts = [{"material": "cotton", "pct": 100}]
    total = sum(p["pct"] for p in parts)
    if total <= 0:
        n = len(parts)
        for p in parts:
            p["pct"] = round(100 / n, 1)
    elif abs(total - 100) > 1:
        scale = 100 / total
        for p in parts:
            p["pct"] = round(p["pct"] * scale, 1)
    return parts


def match_mapping(product: dict, taxonomy: dict) -> dict:
    title = (product.get("title") or "").lower()
    ptype = (product.get("product_type") or "").lower()
    cat = ""
    if product.get("category"):
        cat = (product["category"].get("full_path") or product["category"].get("name") or "").lower()
    tags = [t.lower() for t in (product.get("tags") or [])]
    blob = " ".join([title, ptype, cat, " ".join(tags)])

    for rule in taxonomy["keyword_rules"]:
        if any(kw in blob for kw in rule["keywords"]):
            return dict(rule)

    return dict(taxonomy["default_mapping"])


def infer_season(tags: list[str], mapping: dict, season_map: dict) -> tuple[str, list[str]]:
    tag_blob = " ".join(tags).lower()
    for season, kws in season_map.items():
        if any(kw in tag_blob for kw in kws):
            primary = season
            break
    else:
        primary = mapping.get("season_primary", "ilkbahar")

    usable = {primary}
    thermal = mapping.get("thermal_category", "")
    if thermal in ("kalin_mont", "kaban", "termal_iclik", "kalin_yun_kazak", "bot", "atki", "sapka"):
        usable.add("kis")
        usable.add("sonbahar")
    if thermal in ("ince_pamuklu_tisort", "sort", "yaz_elbisesi"):
        usable.add("yaz")
        usable.add("ilkbahar")
    if "dress" in mapping.get("subcategory", "") or mapping.get("layer_role") == "dress":
        usable.update(["ilkbahar", "yaz"])
    return primary, sorted(usable)


def bucket_product(product: dict, taxonomy: dict) -> str:
    ptype = (product.get("product_type") or "").lower()
    cat = ""
    if product.get("category"):
        cat = (product["category"].get("name") or "").lower()
    blob = f"{ptype} {cat}"
    for bucket, kws in taxonomy["category_bucket_keywords"].items():
        if any(kw in blob for kw in kws):
            return bucket
    return "tops"


def select_products(products: list[dict], taxonomy: dict, target: int, seed: int) -> list[dict]:
    import random

    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {b: [] for b in taxonomy["category_bucket_targets"]}
    for p in products:
        if not p.get("handle"):
            continue
        b = bucket_product(p, taxonomy)
        if b in buckets:
            buckets[b].append(p)

    selected: list[dict] = []
    targets = taxonomy["category_bucket_targets"]
    total_target = sum(targets.values())
    scale = target / total_target

    for bucket, count in targets.items():
        pool = buckets[bucket]
        rng.shuffle(pool)
        n = min(len(pool), max(1, round(count * scale)))
        selected.extend(pool[:n])

    if len(selected) < target:
        used_handles = {p["handle"] for p in selected}
        rest = [p for p in products if p.get("handle") and p["handle"] not in used_handles]
        rng.shuffle(rest)
        selected.extend(rest[: target - len(selected)])

    return selected[:target]


def to_garment(
    product: dict,
    idx: int,
    taxonomy: dict,
    coverage_map: dict,
    snapshot_date: str,
) -> dict | None:
    mapping = match_mapping(product, taxonomy)
    sub = mapping["subcategory"]
    cov_key = mapping.get("coverage_key", sub)
    coverage = coverage_map.get(cov_key, coverage_map.get(sub, 0.70))

    tags = product.get("tags") or []
    season_primary, season_usable = infer_season(tags, mapping, taxonomy["season_tag_keywords"])
    fabrics = parse_fabric(product)
    color_lab = infer_color_lab(product.get("title", ""), tags)
    fabric_desc = ", ".join(f"%{int(f['pct'])} {f['material']}" for f in fabrics)
    title = product.get("title", f"Item {idx}")
    layer = mapping.get("layer_role", "mid")
    desc = (
        f"{title}: {product.get('product_type', 'fashion item')}, "
        f"{fabric_desc}, {layer} katman, Livostyle görsel envanter."
    )

    handle = product["handle"]
    gid = f"LV{idx:04d}"
    image_rel = f"data/assets/garments/{gid}.jpg"

    category_map = {
        "tshirt": "top", "tank_top": "top", "thermal_base": "top",
        "blouse": "top", "shirt": "top", "sweater": "top", "cardigan": "top", "hoodie": "top",
        "jeans": "bottom", "trousers": "bottom", "chinos": "bottom",
        "shorts": "bottom", "skirt": "bottom", "leggings": "bottom",
        "blazer": "outer", "jacket": "outer", "coat": "outer",
        "padded_coat": "outer", "raincoat": "outer",
        "dress_short": "dress", "dress_midi": "dress", "dress_long": "dress",
        "boots": "footwear", "sneakers": "footwear", "sandals": "footwear",
        "scarf": "accessory", "hat": "accessory", "tights": "accessory",
    }

    featured = product.get("featured_image_url") or ""
    if not featured and product.get("images"):
        featured = product["images"][0].get("url", "")

    return {
        "id": gid,
        "name": title,
        "category": category_map.get(sub, "top"),
        "subcategory": sub,
        "layer_role": layer,
        "description": desc,
        "fabric_composition": fabrics,
        "layers": None,
        "sleeve": mapping.get("sleeve"),
        "coverage_ratio": coverage,
        "color_lab": color_lab,
        "season_primary": season_primary,
        "season_usable": season_usable,
        "thermal_category": mapping["thermal_category"],
        "image_path": image_rel,
        "image_source_url": featured,
        "source": "livostyle_mit",
        "source_id": handle,
        "license": "MIT",
        "snapshot_date": snapshot_date,
        "active": True,
        "livostyle_url": product.get("url", ""),
    }


def mirror_images(garments: list[dict], products_by_handle: dict[str, dict]) -> tuple[int, int]:
    ok, fail = 0, 0
    for g in garments:
        handle = g["source_id"]
        gid = g["id"]
        dest = ROOT / g["image_path"]
        if dest.exists() and dest.stat().st_size > 500:
            ok += 1
            continue
        thumb_url = f"{THUMB_BASE}/{handle}.jpg"
        if download_file(thumb_url, dest):
            ok += 1
            continue
        product = products_by_handle.get(handle, {})
        featured = product.get("featured_image_url") or ""
        if featured and download_file(featured, dest):
            ok += 1
        else:
            fail += 1
    return ok, fail


def main() -> None:
    parser = argparse.ArgumentParser(description="Livostyle görsel envanter importu")
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Hedef parça (0 = tüm katalog)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refresh", action="store_true", help="Mevcut manifest üzerine yaz")
    parser.add_argument("--skip-images", action="store_true", help="Sadece metadata JSON")
    args = parser.parse_args()

    taxonomy = load_json(LOOKUPS / "livostyle_taxonomy_map.json")
    coverage_data = load_json(LOOKUPS / "coverage_ratios.json")
    coverage_map = coverage_data["coverage_by_subcategory"]

    print("Livostyle katalog indiriliyor...")
    products = fetch_json(PRODUCTS_URL)
    if not isinstance(products, list):
        print("Beklenmeyen katalog formatı", file=sys.stderr)
        sys.exit(1)
    print(f"  Toplam ürün: {len(products)}")

    if args.target <= 0:
        selected = [p for p in products if p.get("handle")]
        print(f"  Mod: TÜM KATALOG ({len(selected)} parça)")
    else:
        selected = select_products(products, taxonomy, args.target, args.seed)
        print(f"  Mod: örneklem ({len(selected)} parça)")

    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    products_by_handle = {p["handle"]: p for p in products if p.get("handle")}

    garments = []
    for i, product in enumerate(selected, start=1):
        g = to_garment(product, i, taxonomy, coverage_map, snapshot_date)
        if g:
            garments.append(g)

    if args.skip_images:
        ok, fail = 0, 0
        print("  Görseller atlandı (--skip-images)")
    else:
        ok, fail = mirror_images(garments, products_by_handle)
        print(f"  Görseller: {ok} OK, {fail} başarısız")

    VISUAL.mkdir(parents=True, exist_ok=True)
    out = {
        "version": "1.0",
        "source": "livostyle_mit",
        "license": "MIT",
        "count": len(garments),
        "snapshot_date": snapshot_date,
        "note": "Tam Livostyle katalog — lokal mirror, hotlink yok",
        "garments": garments,
    }
    out_path = VISUAL / "garments_livostyle.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    manifest = {
        "snapshot_date": snapshot_date,
        "source_url": PRODUCTS_URL,
        "license": "MIT",
        "garment_count": len(garments),
        "images_ok": ok,
        "images_failed": fail,
        "output": str(out_path.relative_to(ROOT)),
        "refresh": args.refresh,
    }
    with (VISUAL / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"Yazıldı: {out_path}")
    print(f"Manifest: {VISUAL / 'manifest.json'}")


if __name__ == "__main__":
    main()
