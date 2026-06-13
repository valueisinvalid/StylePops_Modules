#!/usr/bin/env python3
"""
Livostyle üretim gardırobunu temizler + 44K'dan kış/sonbahar takviye ekler.
Çıktı: data/visual/garments_production.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "data" / "visual"
REGISTRY = VISUAL / "inventory_registry.json"
LIVO_PATH = VISUAL / "garments_livostyle.json"
FP_PATH = VISUAL / "garments_fashion_product.json"
OUT_PATH = VISUAL / "garments_production.json"

sys.path.insert(0, str(ROOT / "scripts"))
from garment_eligibility import exclusion_reason, is_catalog_eligible


def load_garments_list(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("garments", data if isinstance(data, list) else [])


def clean_livostyle(garments: list[dict]) -> tuple[list[dict], Counter]:
    kept, reasons = [], Counter()
    for g in garments:
        reason = exclusion_reason(g)
        if reason:
            reasons[reason.split(":")[0]] += 1
            continue
        if not is_catalog_eligible(g):
            reasons["missing_image"] += 1
            continue
        kept.append(dict(g))
    return kept, reasons


OUTER_NAME_KEYWORDS = (
    "jacket", "coat", "blazer", "parka", "trench", "puffer", "windbreaker",
    "raincoat", "rain jacket", "overcoat", "peacoat", "bomber", "anorak",
)
MID_WARM_KEYWORDS = ("sweater", "hoodie", "cardigan", "pullover", "fleece", "sweatshirt")


def supplement_layer_role(g: dict) -> str:
    """44K import'ta ceket/mont 'mid' olarak etiketli; takviyede katmanı yeniden ata."""
    blob = f"{g.get('name', '')} {g.get('description', '')}".lower()
    layer = g.get("layer_role", "")
    if layer == "footwear":
        return "footwear"
    if layer == "bottom":
        return "bottom"
    if layer == "dress":
        return "dress"
    if any(k in blob for k in OUTER_NAME_KEYWORDS):
        return "outer"
    if any(k in blob for k in MID_WARM_KEYWORDS):
        return "mid"
    return layer or "mid"


def fp_seasonal_score(g: dict) -> int:
    blob = f"{g.get('name', '')} {g.get('description', '')}".lower()
    sub = g.get("subcategory", "")
    layer = supplement_layer_role(g)
    season = g.get("season_primary", "")
    score = 0
    if season in ("kis", "sonbahar"):
        score += 4
    if "kis" in g.get("season_usable", []):
        score += 2
    if "sonbahar" in g.get("season_usable", []):
        score += 1
    if layer == "outer":
        score += 6
    if sub in ("coat", "padded_coat", "raincoat", "boots"):
        score += 5
    if sub == "sweater":
        score += 3
    if sub in ("jeans", "chinos", "trousers"):
        score += 1
    if any(k in blob for k in ("wool", "fleece", "puffer", "parka", "trench", "winter", "thermal")):
        score += 3
    if any(k in blob for k in ("boot", "chelsea", "ankle boot")):
        score += 4
    if any(k in blob for k in ("raincoat", "rain jacket", "windbreaker", "waterproof")):
        score += 5
    return score


def pick_fp_supplements(
    fp_garments: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    candidates = []
    for g in fp_garments:
        if g.get("season_primary") not in ("kis", "sonbahar") and "kis" not in g.get("season_usable", []):
            continue
        if exclusion_reason(g):
            continue
        if not is_catalog_eligible(g):
            continue
        if g.get("layer_role") == "accessory":
            continue
        score = fp_seasonal_score(g)
        if score < 4:
            continue
        candidates.append((score, g))

    candidates.sort(key=lambda x: (-x[0], x[1]["id"]))
    quotas = {
        "outer": max(30, n // 3),
        "footwear": max(18, n // 5),
        "mid": max(20, n // 4),
        "bottom": max(12, n // 8),
        "dress": 5,
    }
    picked: list[dict] = []
    used_ids: set[str] = set()

    def try_pick(layer_key: str) -> None:
        nonlocal picked
        for score, g in candidates:
            if len(picked) >= n:
                return
            if g["id"] in used_ids:
                continue
            layer = supplement_layer_role(g)
            if layer != layer_key:
                continue
            if sum(1 for p in picked if supplement_layer_role(p) == layer_key) >= quotas.get(layer_key, 99):
                continue
            used_ids.add(g["id"])
            picked.append(g)

    for key in ("outer", "footwear", "mid", "bottom"):
        try_pick(key)

    for score, g in candidates:
        if len(picked) >= n:
            break
        if g["id"] in used_ids:
            continue
        used_ids.add(g["id"])
        picked.append(g)

    rng.shuffle(picked)
    out = []
    for i, g in enumerate(picked[:n], 1):
        item = dict(g)
        item["id"] = f"SP{i:04d}"
        item["layer_role"] = supplement_layer_role(g)
        item["source"] = "fashion_product_supplement"
        item["training_only"] = False
        item["supplement_from_fp"] = g["id"]
        item["active"] = True
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Üretim gardırobu temizlik + 44K takviye")
    parser.add_argument("--supplement", type=int, default=100, help="44K kış/sonbahar parça sayısı")
    parser.add_argument("--no-supplement", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    livo = load_garments_list(LIVO_PATH)
    if not livo:
        print(f"Livostyle bulunamadı: {LIVO_PATH}", file=sys.stderr)
        sys.exit(1)

    cleaned, reasons = clean_livostyle(livo)
    print(f"Livostyle: {len(livo)} → {len(cleaned)} (elenen {len(livo) - len(cleaned)})")
    print("Elenme nedenleri:", dict(reasons))

    supplements: list[dict] = []
    if not args.no_supplement and args.supplement > 0:
        fp = load_garments_list(FP_PATH)
        if not fp:
            print(f"Uyarı: 44K metadata yok ({FP_PATH}), takviye atlandı")
        else:
            supplements = pick_fp_supplements(fp, args.supplement, args.seed)
            print(f"44K takviye: {len(supplements)} parça")
            by_layer = Counter(supplement_layer_role(g) for g in supplements)
            print("  katman:", dict(by_layer))

    merged = cleaned + supplements
    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "version": "1.0",
        "source": "livostyle_mit_curated",
        "license": "MIT",
        "count": len(merged),
        "snapshot_date": snapshot,
        "note": "Livostyle temizlenmiş + Fashion Product 44K kış/sonbahar takviye (SP*)",
        "curation": {
            "livostyle_in": len(livo),
            "livostyle_kept": len(cleaned),
            "excluded_reasons": dict(reasons),
            "fp_supplement": len(supplements),
        },
        "garments": merged,
    }
    VISUAL.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Kaydedildi → {OUT_PATH} ({len(merged)} parça)")

    if REGISTRY.exists():
        reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    else:
        reg = {}
    reg["production_wardrobe"] = "garments_production.json"
    reg.setdefault("notes", {})["production_wardrobe"] = (
        "Temizlenmiş Livostyle + 44K kış/sonbahar takviye (SP*)"
    )
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Registry güncellendi → {REGISTRY}")


if __name__ == "__main__":
    main()
