#!/usr/bin/env python3
"""Bootstrap envanter (200 parça) ve etiketleme CSV (200 kombin) üreticisi."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LOOKUPS = DATA / "lookups"
BOOTSTRAP = DATA / "bootstrap"

CATEGORY_MAP = {
    "tshirt": "top", "tank_top": "top", "blouse": "top", "shirt": "top",
    "sweater": "top", "cardigan": "top", "hoodie": "top",
    "jeans": "bottom", "trousers": "bottom", "chinos": "bottom",
    "shorts": "bottom", "skirt": "bottom", "leggings": "bottom",
    "blazer": "outer", "jacket": "outer", "coat": "outer",
    "padded_coat": "outer", "raincoat": "outer",
    "dress_short": "dress", "dress_midi": "dress", "dress_long": "dress",
    "boots": "footwear", "sneakers": "footwear", "sandals": "footwear",
    "scarf": "accessory",
}

OUTER_SUBCATEGORIES = {"blazer", "jacket", "coat", "padded_coat", "raincoat", "cardigan"}
TOP_SUBCATEGORIES = {"tshirt", "tank_top", "blouse", "shirt", "sweater", "hoodie"}
BOTTOM_SUBCATEGORIES = {"jeans", "trousers", "chinos", "shorts", "skirt", "leggings"}
DRESS_SUBCATEGORIES = {"dress_short", "dress_midi", "dress_long"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def lab_to_description(L: float, a: float, b: float) -> str:
    if L > 80:
        tone = "açık"
    elif L > 55:
        tone = "orta"
    else:
        tone = "koyu"
    if b > 15:
        hue = "sıcak"
    elif b < -5:
        hue = "soğuk"
    else:
        hue = "nötr"
    return f"{tone} {hue} tonlu"


def fabric_description(fabrics: list[dict]) -> str:
    parts = [f"%{int(f['pct'])} {f['material']}" for f in fabrics]
    return ", ".join(parts)


def weighted_ic_clo(fabrics: list[dict], materials: dict) -> float:
    total_pct = sum(f["pct"] for f in fabrics)
    if total_pct == 0:
        return 0.0
    return sum(
        (f["pct"] / total_pct) * materials[f["material"]]["clo"]
        for f in fabrics
    )


def compute_effective_clo(
    garment: dict,
    materials: dict,
    thermal_categories: dict,
    use_category_fallback: bool = False,
) -> float:
    if use_category_fallback:
        cat = thermal_categories[garment["thermal_category"]]
        ic_clo = cat["clo"]
    else:
        ic_clo = weighted_ic_clo(garment["fabric_composition"], materials)
    return round(ic_clo * garment["coverage_ratio"], 4)


def wind_chill_c(T_c: float, V_kmh: float) -> float:
    if T_c > 10 or V_kmh < 4.8:
        return T_c
    v_ms = V_kmh / 3.6
    return 13.12 + 0.6215 * T_c - 11.37 * (v_ms ** 0.16) + 0.3965 * T_c * (v_ms ** 0.16)


def heat_index_c(T_c: float, RH: float) -> float:
    if T_c < 27:
        return T_c
    Tf = T_c * 9 / 5 + 32
    HI = (
        -42.379 + 2.04901523 * Tf + 10.14333127 * RH
        - 0.22475541 * Tf * RH - 0.00683783 * Tf ** 2
        - 0.05481717 * RH ** 2 + 0.00122874 * Tf ** 2 * RH
        + 0.00085282 * Tf * RH ** 2 - 0.00000199 * Tf ** 2 * RH ** 2
    )
    return (HI - 32) * 5 / 9


def apparent_temperature(T_hava: float, RH: float, V_ruzgar: float) -> float:
    if T_hava <= 10 and V_ruzgar >= 5:
        return round(wind_chill_c(T_hava, V_ruzgar), 2)
    if T_hava >= 27:
        return round(heat_index_c(T_hava, RH), 2)
    return round(T_hava, 2)


def interpolate_hedef_clo(T: float, points: list[dict]) -> float:
    sorted_pts = sorted(points, key=lambda p: p["T_celsius"])
    if T <= sorted_pts[0]["T_celsius"]:
        return sorted_pts[0]["clo"]
    if T >= sorted_pts[-1]["T_celsius"]:
        return sorted_pts[-1]["clo"]
    for i in range(len(sorted_pts) - 1):
        t1, c1 = sorted_pts[i]["T_celsius"], sorted_pts[i]["clo"]
        t2, c2 = sorted_pts[i + 1]["T_celsius"], sorted_pts[i + 1]["clo"]
        if t1 <= T <= t2:
            if t2 == t1:
                return c1
            return round(c1 + (T - t1) * (c2 - c1) / (t2 - t1), 4)
    return sorted_pts[-1]["clo"]


def generate_garments(seed: int = 42) -> list[dict]:
    random.seed(seed)
    season_data = load_json(LOOKUPS / "season_templates.json")
    coverage_data = load_json(LOOKUPS / "coverage_ratios.json")
    coverage_map = coverage_data["coverage_by_subcategory"]
    sleeve_map = coverage_data["coverage_by_sleeve"]
    usable_map = season_data["season_usable_map"]

    garments: list[dict] = []
    gid = 1

    for season_key, season_info in season_data["seasons"].items():
        templates = season_info["templates"]
        count = season_info["target_count"]
        per_template = count // len(templates)
        remainder = count % len(templates)

        for t_idx, template in enumerate(templates):
            n = per_template + (1 if t_idx < remainder else 0)
            for v in range(n):
                color = random.choice(template["color_palette"])
                L, a, b = color
                fabrics = template["fabrics"]
                sub = template["subcategory"]
                sleeve = template.get("sleeve")
                coverage = coverage_map.get(sub, 0.70)
                if sleeve and sub in TOP_SUBCATEGORIES | {"blouse", "shirt", "cardigan", "sweater"}:
                    coverage = max(coverage, sleeve_map.get(sleeve, coverage))

                name = f"{template['name_prefix']} {v + 1:02d}"
                desc_color = lab_to_description(L, a, b)
                desc_fabric = fabric_description(fabrics)
                description = (
                    f"{name}: {desc_color} renkte, {desc_fabric} kumaştan "
                    f"{season_info['label_tr'].lower()} parçası."
                )

                layers = None
                if sub in {"padded_coat", "coat", "jacket"}:
                    layers = [
                        {
                            "role": "shell",
                            "weight": 0.70,
                            "fabric_composition": fabrics,
                        },
                        {
                            "role": "fill" if sub == "padded_coat" else "lining",
                            "weight": 0.30,
                            "fabric_composition": (
                                [{"material": "down", "pct": 100}]
                                if sub == "padded_coat"
                                else [{"material": "polyester", "pct": 100}]
                            ),
                        },
                    ]

                garment = {
                    "id": f"G{gid:03d}",
                    "name": name,
                    "category": CATEGORY_MAP[sub],
                    "subcategory": sub,
                    "description": description,
                    "fabric_composition": fabrics,
                    "layers": layers,
                    "sleeve": sleeve,
                    "coverage_ratio": round(coverage, 3),
                    "color_lab": {"L": L, "a": a, "b": b},
                    "season_primary": season_key,
                    "season_usable": usable_map[season_key],
                    "thermal_category": template["thermal_category"],
                }
                garments.append(garment)
                gid += 1

    return garments


def pick_combo_items(
    pool: list[dict],
    season: str,
    need_outer: bool,
    rng: random.Random,
) -> tuple[dict | None, dict | None, dict | None]:
    tops = [g for g in pool if g["subcategory"] in TOP_SUBCATEGORIES and season in g["season_usable"]]
    bottoms = [g for g in pool if g["subcategory"] in BOTTOM_SUBCATEGORIES and season in g["season_usable"]]
    outers = [g for g in pool if g["subcategory"] in OUTER_SUBCATEGORIES and season in g["season_usable"]]
    dresses = [g for g in pool if g["subcategory"] in DRESS_SUBCATEGORIES and season in g["season_usable"]]

    if dresses and rng.random() < 0.15:
        dress = rng.choice(dresses)
        outer = rng.choice(outers) if need_outer and outers else None
        return dress, None, outer

    if not tops or not bottoms:
        return None, None, None

    top = rng.choice(tops)
    bottom = rng.choice(bottoms)
    outer = rng.choice(outers) if need_outer and outers and rng.random() < 0.9 else None
    return top, bottom, outer


def generate_combinations(garments: list[dict], seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    season_data = load_json(LOOKUPS / "season_templates.json")
    target_data = load_json(LOOKUPS / "target_clo_points.json")
    fabric_data = load_json(LOOKUPS / "fabric_properties.json")
    materials = fabric_data["materials"]
    thermal_categories = fabric_data["thermal_categories"]
    clo_points = target_data["target_clo_points"]
    scenarios = target_data["weather_scenarios"]
    combo_rules = season_data["combo_rules"]

    rows: list[dict] = []
    combo_id = 1

    for season_key, rules in combo_rules.items():
        season_scenarios = [scenarios[s] for s in rules["scenarios"]]
        combos_per_season = 50

        for i in range(combos_per_season):
            scenario_key = rules["scenarios"][i % len(rules["scenarios"])]
            scenario = scenarios[scenario_key]
            need_outer = rng.random() < rules["outer_probability"]
            top, bottom, outer = pick_combo_items(garments, season_key, need_outer, rng)
            if top is None:
                continue

            piece_ids = []
            total_clo = 0.0
            for piece in [top, bottom, outer]:
                if piece:
                    piece_ids.append(piece["id"])
                    total_clo += compute_effective_clo(
                        piece, materials, thermal_categories
                    )

            T_hava = scenario["T_hava"]
            RH = scenario["RH_nem"]
            V = scenario["V_ruzgar"]
            T_hissedilen = apparent_temperature(T_hava, RH, V)
            hedef_clo = interpolate_hedef_clo(T_hissedilen, clo_points)
            delta_clo = round(abs(hedef_clo - total_clo), 4)

            if top["category"] == "dress":
                combo_type = "dress+outer" if outer else "dress"
                top_id, bottom_id = top["id"], ""
            else:
                combo_type = "top+bottom+outer" if outer else "top+bottom"
                top_id = top["id"]
                bottom_id = bottom["id"] if bottom else ""

            rows.append({
                "combo_id": f"C{combo_id:03d}",
                "season": season_key,
                "combo_type": combo_type,
                "top_id": top_id,
                "bottom_id": bottom_id,
                "outer_id": outer["id"] if outer else "",
                "piece_ids": "|".join(piece_ids),
                "weather_scenario": scenario_key,
                "weather_label_tr": scenario.get("label_tr", ""),
                "T_hava": T_hava,
                "RH_nem": RH,
                "V_ruzgar": V,
                "T_hissedilen": T_hissedilen,
                "hedef_Clo": hedef_clo,
                "total_Clo_C": round(total_clo, 4),
                "delta_Clo": delta_clo,
                "aesthetic_score": "",
                "thermal_score": "",
                "labeler": "",
                "notes": "",
            })
            combo_id += 1

    return rows


def write_garments(garments: list[dict]) -> None:
    BOOTSTRAP.mkdir(parents=True, exist_ok=True)
    out = BOOTSTRAP / "garments_200.json"
    payload = {
        "version": "1.0",
        "count": len(garments),
        "note": "Bootstrap envanter — curated synthetic, gerçek gardırop dağılımına uygun",
        "garments": garments,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(garments)} garments -> {out}")


def write_combinations(rows: list[dict]) -> None:
    out = BOOTSTRAP / "combinations_200.csv"
    fieldnames = [
        "combo_id", "season", "combo_type",
        "top_id", "bottom_id", "outer_id", "piece_ids",
        "weather_scenario", "weather_label_tr",
        "T_hava", "RH_nem", "V_ruzgar", "T_hissedilen",
        "hedef_Clo", "total_Clo_C", "delta_Clo",
        "aesthetic_score", "thermal_score", "labeler", "notes",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} combinations -> {out}")


def main() -> None:
    garments = generate_garments()
    assert len(garments) == 200, f"Expected 200 garments, got {len(garments)}"

    season_counts: dict[str, int] = {}
    for g in garments:
        season_counts[g["season_primary"]] = season_counts.get(g["season_primary"], 0) + 1
    print("Season distribution:", season_counts)

    write_garments(garments)

    combos = generate_combinations(garments)
    assert len(combos) == 200, f"Expected 200 combinations, got {len(combos)}"

    combo_season_counts: dict[str, int] = {}
    for c in combos:
        combo_season_counts[c["season"]] = combo_season_counts.get(c["season"], 0) + 1
    print("Combo season distribution:", combo_season_counts)

    write_combinations(combos)


if __name__ == "__main__":
    main()
