#!/usr/bin/env python3
"""
Livostyle üretim gardırobunu temizler + 44K'dan filtreli takviye ekler.
Çıktı: data/visual/garments_production.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
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

SEASONS = ("kis", "sonbahar", "ilkbahar", "yaz")

OUTER_NAME_KEYWORDS = (
    "jacket", "coat", "blazer", "parka", "trench", "puffer", "windbreaker",
    "raincoat", "rain jacket", "overcoat", "peacoat", "bomber", "anorak",
)
MID_WARM_KEYWORDS = ("sweater", "hoodie", "cardigan", "pullover", "fleece", "sweatshirt")
SUMMER_KEYWORDS = ("linen", "cotton", "short sleeve", "sleeveless", "tank", "lightweight")
WINTER_KEYWORDS = ("wool", "fleece", "puffer", "parka", "trench", "winter", "thermal", "down jacket")


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
    if layer == "base":
        return "base"
    return layer or "mid"


def is_valid_fp_supplement(g: dict) -> bool:
    from garment_eligibility import non_garment_reason
    from stylepops_core import is_beach_swim_garment, is_valid_bottom_piece

    if non_garment_reason(g):
        return False
    layer = supplement_layer_role(g)
    if layer not in {"outer", "footwear", "bottom", "mid", "base", "dress"}:
        return False
    if is_beach_swim_garment(g):
        return False
    if layer == "bottom":
        probe = dict(g)
        probe["layer_role"] = "bottom"
        if not is_valid_bottom_piece(probe):
            return False
    return True


def fp_production_score(g: dict) -> int:
    blob = f"{g.get('name', '')} {g.get('description', '')}".lower()
    sub = g.get("subcategory", "")
    layer = supplement_layer_role(g)
    season = g.get("season_primary", "")
    score = 1

    if layer == "outer":
        score += 6
    elif layer == "footwear":
        score += 4
    elif layer == "bottom":
        score += 3
    elif layer == "mid":
        score += 2
    elif layer == "base":
        score += 2

    if season in SEASONS:
        score += 2
    if season == "kis" and any(k in blob for k in WINTER_KEYWORDS):
        score += 4
    if season == "sonbahar" and any(k in blob for k in WINTER_KEYWORDS + ("raincoat", "windbreaker")):
        score += 3
    if season in ("ilkbahar", "yaz") and any(k in blob for k in SUMMER_KEYWORDS):
        score += 2
    if sub in ("boots", "coat", "padded_coat", "raincoat", "jeans", "chinos", "trousers"):
        score += 3
    if any(k in blob for k in ("shacket", "fleece shacket")):
        score -= 2
    return score


def layer_quotas(n: int) -> dict[str, int]:
    raw = {
        "outer": 0.22,
        "footwear": 0.14,
        "bottom": 0.14,
        "mid": 0.28,
        "base": 0.12,
        "dress": 0.05,
    }
    quotas = {k: max(1, int(n * pct)) for k, pct in raw.items()}
    delta = n - sum(quotas.values())
    quotas["mid"] += delta
    return quotas


def pick_fp_supplements(
    fp_garments: list[dict],
    n: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_layer: dict[str, list[tuple[int, dict]]] = defaultdict(list)

    for g in fp_garments:
        if exclusion_reason(g):
            continue
        if not is_catalog_eligible(g):
            continue
        if g.get("layer_role") == "accessory":
            continue
        if not is_valid_fp_supplement(g):
            continue
        score = fp_production_score(g)
        if score < 4:
            continue
        layer = supplement_layer_role(g)
        by_layer[layer].append((score, g))

    for layer in by_layer:
        by_layer[layer].sort(key=lambda x: (-x[0], x[1]["id"]))

    quotas = layer_quotas(n)
    picked: list[dict] = []
    used_ids: set[str] = set()
    season_counts: Counter = Counter()

    def season_of(g: dict) -> str:
        s = g.get("season_primary", "")
        return s if s in SEASONS else "ilkbahar"

    target_per_season = max(80, n // len(SEASONS))

    for layer, quota in quotas.items():
        pool = by_layer.get(layer, [])
        if not pool:
            continue
        layer_picked = 0
        by_season_pool: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for score, g in pool:
            by_season_pool[season_of(g)].append((score, g))
        for items in by_season_pool.values():
            items.sort(key=lambda x: (-x[0], x[1]["id"]))

        season_order = list(SEASONS)
        rng.shuffle(season_order)
        idx = {s: 0 for s in SEASONS}
        guard = 0
        while layer_picked < quota and guard < quota * 20:
            guard += 1
            for s in season_order:
                if layer_picked >= quota:
                    break
                items = by_season_pool.get(s, [])
                while idx[s] < len(items) and items[idx[s]][1]["id"] in used_ids:
                    idx[s] += 1
                if idx[s] >= len(items):
                    continue
                score, g = items[idx[s]]
                idx[s] += 1
                used_ids.add(g["id"])
                picked.append(g)
                season_counts[s] += 1
                layer_picked += 1

    if len(picked) < n:
        rest = []
        for layer_items in by_layer.values():
            for score, g in layer_items:
                if g["id"] not in used_ids:
                    rest.append((score, g))
        rest.sort(key=lambda x: (-x[0], x[1]["id"]))
        for score, g in rest:
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
    parser.add_argument("--supplement", type=int, default=1500, help="44K filtreli takviye parça sayısı")
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
            by_layer = Counter(g["layer_role"] for g in supplements)
            by_season = Counter(g.get("season_primary", "?") for g in supplements)
            print("  katman:", dict(by_layer))
            print("  mevsim:", dict(by_season))

    merged = cleaned + supplements
    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "version": "1.0",
        "source": "livostyle_mit_curated",
        "license": "MIT",
        "count": len(merged),
        "snapshot_date": snapshot,
        "note": f"Livostyle temizlenmiş + Fashion Product 44K filtreli takviye ({len(supplements)} SP*)",
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
        f"Temizlenmiş Livostyle + 44K filtreli takviye ({len(supplements)} SP*)"
    )
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Registry güncellendi → {REGISTRY}")


if __name__ == "__main__":
    main()
