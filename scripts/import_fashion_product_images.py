#!/usr/bin/env python3
"""
Fashion Product Images Small (~44K, MIT) importu.
Görselleri lokal mirror eder; estetik eğitim korpusudur (üretim gardırobu değil).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOOKUPS = ROOT / "data" / "lookups"
VISUAL = ROOT / "data" / "visual"
ASSETS = ROOT / "data" / "assets" / "fashion_product"
PROGRESS_PATH = VISUAL / "fashion_product_import_progress.json"
OUT_PATH = VISUAL / "garments_fashion_product.json"
DATASET_ID = "benitomartin/fashion-product-images-small-384x512"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def save_progress(data: dict) -> None:
    VISUAL.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def map_row(row: dict, seq: int, taxonomy: dict, coverage_map: dict, snapshot_date: str) -> dict:
    article = (row.get("articleType") or "").lower()
    sub_cat = (row.get("subCategory") or "").lower()
    blob = f"{article} {sub_cat}"

    mapping = dict(taxonomy["default"])
    for rule in taxonomy["article_type_rules"]:
        if any(kw in blob for kw in rule["keywords"]):
            mapping.update(rule)
            break

    sub = mapping["subcategory"]
    cov_key = mapping.get("coverage_key", sub)
    coverage = coverage_map.get(cov_key, coverage_map.get(sub, 0.70))

    season_raw = (row.get("season") or "Spring").lower()
    season_primary = taxonomy["season_map"].get(season_raw, "ilkbahar")
    season_usable = sorted({season_primary, "ilkbahar", "yaz"} if season_primary in ("ilkbahar", "yaz") else {season_primary, "kis", "sonbahar"})

    colour = (row.get("baseColour") or "").lower()
    lab = taxonomy["colour_lab"].get(colour, [60.0, 5.0, 8.0])
    color_lab = {"L": lab[0], "a": lab[1], "b": lab[2]}

    name = row.get("productDisplayName") or f"Product {row.get('id')}"
    usage = row.get("usage") or "Casual"
    gender = row.get("gender") or ""
    desc = (
        f"{name}: {row.get('articleType')} {usage} {gender}, "
        f"{row.get('baseColour')} colour, Fashion Product MIT training corpus."
    )

    source_id = str(row.get("id"))
    gid = f"FP{seq:06d}"
    image_rel = f"data/assets/fashion_product/{source_id}.jpg"

    return {
        "id": gid,
        "name": name,
        "category": mapping.get("category", "top"),
        "subcategory": sub,
        "layer_role": mapping.get("layer_role", "mid"),
        "description": desc,
        "fabric_composition": mapping.get("fabric_composition", taxonomy["default"]["fabric_composition"]),
        "layers": None,
        "sleeve": mapping.get("sleeve"),
        "coverage_ratio": coverage,
        "color_lab": color_lab,
        "season_primary": season_primary,
        "season_usable": season_usable,
        "thermal_category": mapping["thermal_category"],
        "image_path": image_rel,
        "image_source_url": f"hf://{DATASET_ID}/{source_id}",
        "source": "fashion_product_images_mit",
        "source_id": source_id,
        "license": "MIT",
        "snapshot_date": snapshot_date,
        "active": True,
        "training_only": True,
        "fp_meta": {
            "gender": gender,
            "masterCategory": row.get("masterCategory"),
            "subCategory": row.get("subCategory"),
            "articleType": row.get("articleType"),
            "usage": usage,
            "year": row.get("year"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = tüm 44K")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-log", type=int, default=500)
    args = parser.parse_args()

    from hf_env import ensure_hf_token
    if ensure_hf_token():
        print("HF token yüklendi (.env)")
    else:
        print("Uyarı: HF_TOKEN yok — indirme yavaş olabilir (.env.example)")

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets", file=sys.stderr)
        sys.exit(1)

    taxonomy = load_json(LOOKUPS / "fashion_product_taxonomy_map.json")
    coverage_map = load_json(LOOKUPS / "coverage_ratios.json")["coverage_by_subcategory"]
    snapshot_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    garments: list[dict] = []
    start_idx = 0
    if args.resume and OUT_PATH.exists():
        existing = load_json(OUT_PATH)
        garments = existing.get("garments", [])
        start_idx = len(garments)
        print(f"Resume: {start_idx} parça mevcut")

    print(f"HF dataset yükleniyor: {DATASET_ID}")
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    ASSETS.mkdir(parents=True, exist_ok=True)
    imported = 0
    skipped = 0
    target_total = args.limit if args.limit > 0 else 44072

    for i, row in enumerate(ds):
        if i < start_idx:
            continue
        if args.limit > 0 and imported >= args.limit:
            break
        if imported + start_idx >= target_total and args.limit == 0 and start_idx == 0:
            # full import — continue until dataset ends
            pass

        source_id = str(row["id"])
        dest = ASSETS / f"{source_id}.jpg"
        if not dest.exists():
            try:
                img = row["image"]
                img.convert("RGB").save(dest, format="JPEG", quality=90)
            except Exception:
                skipped += 1
                continue

        garment = map_row(row, start_idx + imported + 1, taxonomy, coverage_map, snapshot_date)
        garments.append(garment)
        imported += 1

        if imported % args.batch_log == 0:
            print(f"  … {start_idx + imported} görsel")
            save_progress({
                "imported": start_idx + imported,
                "skipped": skipped,
                "snapshot_date": snapshot_date,
            })
            # checkpoint catalog
            _write_catalog(garments, snapshot_date)

    _write_catalog(garments, snapshot_date)
    save_progress({
        "imported": len(garments),
        "skipped": skipped,
        "snapshot_date": snapshot_date,
        "complete": True,
    })
    print(f"Tamamlandı: {len(garments)} parça, {skipped} atlandı → {OUT_PATH}")


def _write_catalog(garments: list[dict], snapshot_date: str) -> None:
    VISUAL.mkdir(parents=True, exist_ok=True)
    out = {
        "version": "1.0",
        "source": "fashion_product_images_mit",
        "license": "MIT",
        "count": len(garments),
        "snapshot_date": snapshot_date,
        "training_only": True,
        "note": "44K MIT estetik eğitim korpusu — üretim gardırobunda kullanılmaz",
        "garments": garments,
    }
    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
