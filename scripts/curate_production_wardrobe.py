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
FNAUMAN_PATH = VISUAL / "garments_fnauman.json"
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


import re as _re

# "Dress Pants/Shirt" gibi adlarda "dress" sıfattır; gerçek elbise değil
_BOTTOM_NAME_RE = _re.compile(
    r"\b(pants?|trousers?|jeans|leggings?|joggers?|culottes?|chinos?|"
    r"slacks|sweatpants|track\s*pants|palazzo|cargo)\b",
    _re.I,
)
_REAL_DRESS_RE = _re.compile(
    r"\b(midi|maxi|mini|slip|shirt|bodycon|wrap|gown|sundress|tunic)\s*dress\b|"
    r"\bdress\s*(midi|maxi|mini)\b|\bjumpsuit\b|\bromper\b|\bsundress\b|\bgown\b",
    _re.I,
)


# 44K ayakkabı tipleri hepsi 'sneakers'a çökmüştü; gerçek tipe geri ayır
_FOOTWEAR_ARTICLE_MAP = {
    "Casual Shoes": "loafers",
    "Formal Shoes": "derby",
    "Flats": "flats",
    "Heels": "heels",
    "Sports Shoes": "sneakers",
    "Sandals": "sandals",
    "Sandal": "sandals",
    "Boots": "boots",
}


def correct_labels(item: dict) -> dict:
    """Açıkça yanlış etiketlenmiş katmanları düzelt (ör. 'Dress Pants' → bottom)."""
    name = (item.get("name") or "")
    if item.get("layer_role") == "footwear":
        art = (item.get("fp_meta") or {}).get("articleType")
        if art in _FOOTWEAR_ARTICLE_MAP:
            item["subcategory"] = _FOOTWEAR_ARTICLE_MAP[art]
        else:
            blob = name.lower()
            if "boot" in blob:
                item["subcategory"] = "boots"
            elif "heel" in blob or "pump" in blob or "stiletto" in blob:
                item["subcategory"] = "heels"
            elif "loafer" in blob or "moccasin" in blob:
                item["subcategory"] = "loafers"
            elif "oxford" in blob or "derby" in blob or "brogue" in blob:
                item["subcategory"] = "derby"
            elif "flat" in blob or "ballerina" in blob or "ballet" in blob:
                item["subcategory"] = "flats"
    if item.get("layer_role") == "dress":
        if _BOTTOM_NAME_RE.search(name) and not _REAL_DRESS_RE.search(name):
            item["layer_role"] = "bottom"
            blob = name.lower()
            if "jean" in blob:
                item["subcategory"] = "jeans"
            elif "legging" in blob:
                item["subcategory"] = "leggings"
            elif "jogger" in blob or "sweatpant" in blob or "track" in blob:
                item["subcategory"] = "joggers"
            elif "short" in blob:
                item["subcategory"] = "shorts"
            else:
                item["subcategory"] = "trousers"
    return item


def clean_livostyle(garments: list[dict]) -> tuple[list[dict], Counter]:
    from garment_gender import infer_gender

    kept, reasons = [], Counter()
    for g in garments:
        reason = exclusion_reason(g)
        if reason:
            reasons[reason.split(":")[0]] += 1
            continue
        if not is_catalog_eligible(g):
            reasons["missing_image"] += 1
            continue
        item = dict(g)
        item["gender"] = infer_gender(item)
        correct_labels(item)
        kept.append(item)
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

    from garment_gender import infer_gender as _infer_gender

    def gender_of(g: dict) -> str:
        gd = _infer_gender(g)
        return gd if gd in ("men", "women") else "unisex"

    for layer, quota in quotas.items():
        pool = by_layer.get(layer, [])
        if not pool:
            continue
        # (cinsiyet, mevsim) havuzları — unisex ikisine de uygun
        pools: dict[str, dict[str, list]] = {
            gd: defaultdict(list) for gd in ("men", "women")
        }
        for score, g in pool:
            gd = gender_of(g)
            s = season_of(g)
            if gd == "unisex":
                pools["men"][s].append((score, g))
                pools["women"][s].append((score, g))
            else:
                pools[gd][s].append((score, g))
        for gd in pools:
            for items in pools[gd].values():
                items.sort(key=lambda x: (-x[0], x[1]["id"]))

        # Katman kotasını cinsiyetler arasında eşit böl
        gender_quota = {"men": quota // 2, "women": quota - quota // 2}
        for gd in ("men", "women"):
            gq = gender_quota[gd]
            picked_g = 0
            season_order = list(SEASONS)
            rng.shuffle(season_order)
            idx = {s: 0 for s in SEASONS}
            guard = 0
            while picked_g < gq and guard < gq * 20 + 20:
                guard += 1
                progressed = False
                for s in season_order:
                    if picked_g >= gq:
                        break
                    items = pools[gd].get(s, [])
                    while idx[s] < len(items) and items[idx[s]][1]["id"] in used_ids:
                        idx[s] += 1
                    if idx[s] >= len(items):
                        continue
                    score, g = items[idx[s]]
                    idx[s] += 1
                    used_ids.add(g["id"])
                    picked.append(g)
                    season_counts[s] += 1
                    picked_g += 1
                    progressed = True
                if not progressed:
                    break

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
    from garment_gender import infer_gender

    out = []
    for i, g in enumerate(picked[:n], 1):
        item = dict(g)
        item["id"] = f"SP{i:04d}"
        item["layer_role"] = supplement_layer_role(g)
        item["source"] = "fashion_product_supplement"
        item["training_only"] = False
        item["supplement_from_fp"] = g["id"]
        item["active"] = True
        item["gender"] = infer_gender(item)
        correct_labels(item)
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

    # fnauman (CC-BY 4.0) — kaban/kışlık dış giyim takviyesi
    fnauman_kept: list[dict] = []
    fn_raw = load_garments_list(FNAUMAN_PATH)
    if fn_raw:
        from garment_eligibility import non_garment_reason
        for g in fn_raw:
            if non_garment_reason(g):
                continue
            item = dict(g)
            correct_labels(item)
            fnauman_kept.append(item)
        print(f"fnauman takviye: {len(fnauman_kept)} parça (kaynak {len(fn_raw)})")
        fn_layer = Counter(g["layer_role"] for g in fnauman_kept)
        print("  katman:", dict(fn_layer))

    merged = cleaned + supplements + fnauman_kept
    snapshot = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "version": "1.0",
        "source": "livostyle_mit_curated",
        "license": "MIT + CC-BY-4.0 (fnauman)",
        "count": len(merged),
        "snapshot_date": snapshot,
        "note": (
            f"Livostyle (MIT) temizlenmiş + 44K (MIT) filtreli takviye ({len(supplements)} SP*) "
            f"+ fnauman (CC-BY-4.0) kışlık dış giyim ({len(fnauman_kept)} FN*)"
        ),
        "curation": {
            "livostyle_in": len(livo),
            "livostyle_kept": len(cleaned),
            "excluded_reasons": dict(reasons),
            "fp_supplement": len(supplements),
            "fnauman_supplement": len(fnauman_kept),
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

    from validate_production_wardrobe import main as validate_main
    if validate_main() != 0:
        print("Uyarı: gardırop doğrulaması başarısız — kombin üretmeyin.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
