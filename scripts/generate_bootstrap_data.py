#!/usr/bin/env python3
"""Bootstrap envanter (200 parça) ve etiketleme CSV (200 kombin) üreticisi."""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from stylepops_core import (  # noqa: E402
    SUBCATEGORY_TO_SLOT,
    apparent_temperature,
    ensemble_total_clo,
    generate_layered_candidates,
    interpolate_hedef_clo,
)

DATA = ROOT / "data"
LOOKUPS = DATA / "lookups"
BOOTSTRAP = DATA / "bootstrap"

CATEGORY_MAP = {
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

TOP_SUBCATEGORIES = {"tshirt", "tank_top", "thermal_base", "blouse", "shirt", "sweater", "hoodie"}


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
    return ", ".join(f"%{int(f['pct'])} {f['material']}" for f in fabrics)


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
                cov_key = template.get("coverage_key", sub)
                coverage = coverage_map.get(cov_key, coverage_map.get(sub, 0.70))
                if sleeve and sub in TOP_SUBCATEGORIES | {"blouse", "shirt", "cardigan", "sweater"}:
                    coverage = max(coverage, sleeve_map.get(sleeve, coverage))

                layer_role = template.get("layer_role", SUBCATEGORY_TO_SLOT.get(sub, "mid"))
                name = f"{template['name_prefix']} {v + 1:02d}"
                description = (
                    f"{name}: {lab_to_description(L, a, b)} renkte, {fabric_description(fabrics)} "
                    f"kumaştan {season_info['label_tr'].lower()} parçası ({layer_role} katman)."
                )

                layers = None
                if sub in {"padded_coat", "coat", "jacket"}:
                    layers = [
                        {"role": "shell", "weight": 0.70, "fabric_composition": fabrics},
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

                garments.append({
                    "id": f"G{gid:03d}",
                    "name": name,
                    "category": CATEGORY_MAP[sub],
                    "subcategory": sub,
                    "layer_role": layer_role,
                    "description": description,
                    "fabric_composition": fabrics,
                    "layers": layers,
                    "sleeve": sleeve,
                    "coverage_ratio": round(coverage, 3),
                    "color_lab": {"L": L, "a": a, "b": b},
                    "season_primary": season_key,
                    "season_usable": usable_map[season_key],
                    "thermal_category": template["thermal_category"],
                })
                gid += 1

    return garments


def generate_combinations(garments: list[dict], seed: int = 42) -> list[dict]:
    season_data = load_json(LOOKUPS / "season_templates.json")
    target_data = load_json(LOOKUPS / "target_clo_points.json")
    fabric_data = load_json(LOOKUPS / "fabric_properties.json")
    coverage_data = load_json(LOOKUPS / "coverage_ratios.json")
    thermal_categories = fabric_data["thermal_categories"]
    coverage_defaults = coverage_data["coverage_by_subcategory"]
    clo_points = target_data["target_clo_points"]
    scenarios = target_data["weather_scenarios"]
    combo_rules = season_data["combo_rules"]

    garment_dict = {g["id"]: g for g in garments}
    rows: list[dict] = []
    combo_id = 1

    for season_key, rules in combo_rules.items():
        combos_per_season = 50
        for i in range(combos_per_season):
            scenario_key = rules["scenarios"][i % len(rules["scenarios"])]
            scenario = scenarios[scenario_key]
            T_hava = scenario["T_hava"]
            RH = scenario["RH_nem"]
            V = scenario["V_ruzgar"]
            T_hissedilen = apparent_temperature(T_hava, RH, V)
            hedef_clo = interpolate_hedef_clo(T_hissedilen, clo_points)

            candidates = generate_layered_candidates(
                garment_dict,
                n_candidates=30,
                season=season_key,
                hedef_clo=hedef_clo,
                V_ruzgar=V,
                seed=seed + combo_id,
            )
            if not candidates:
                continue
            piece_ids = candidates[i % len(candidates)]
            pieces = [garment_dict[pid] for pid in piece_ids]
            total_clo = ensemble_total_clo(pieces, thermal_categories, coverage_defaults)

            by_slot = {}
            for p in pieces:
                by_slot.setdefault(p["layer_role"], []).append(p["id"])

            rows.append({
                "combo_id": f"C{combo_id:03d}",
                "season": season_key,
                "combo_type": f"layered_{len(piece_ids)}p",
                "base_id": "|".join(by_slot.get("base", [])),
                "mid_id": "|".join(by_slot.get("mid", [])),
                "outer_id": "|".join(by_slot.get("outer", [])),
                "bottom_id": "|".join(by_slot.get("bottom", [])),
                "accessory_ids": "|".join(by_slot.get("accessory", [])),
                "top_id": "|".join(by_slot.get("base", []) + by_slot.get("mid", [])),
                "piece_ids": "|".join(piece_ids),
                "layer_count": len(piece_ids),
                "weather_scenario": scenario_key,
                "weather_label_tr": scenario.get("label_tr", ""),
                "T_hava": T_hava,
                "RH_nem": RH,
                "V_ruzgar": V,
                "T_hissedilen": T_hissedilen,
                "hedef_Clo": hedef_clo,
                "total_Clo_C": total_clo,
                "delta_Clo": round(abs(hedef_clo - total_clo), 4),
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
    with out.open("w", encoding="utf-8") as f:
        json.dump({
            "version": "1.1",
            "count": len(garments),
            "note": "Katmanlı giyim destekli bootstrap envanter",
            "garments": garments,
        }, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(garments)} garments -> {out}")


def write_combinations(rows: list[dict]) -> None:
    out = BOOTSTRAP / "combinations_200.csv"
    fieldnames = [
        "combo_id", "season", "combo_type", "layer_count",
        "base_id", "mid_id", "outer_id", "bottom_id", "accessory_ids",
        "top_id", "piece_ids",
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
    print("Garment count:", len(garments))
    print("Season distribution:", {
        s: sum(1 for g in garments if g["season_primary"] == s)
        for s in {"kis", "sonbahar", "ilkbahar", "yaz"}
    })
    write_garments(garments)

    combos = generate_combinations(garments)
    print("Combo count:", len(combos))
    write_combinations(combos)


if __name__ == "__main__":
    main()
