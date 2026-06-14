#!/usr/bin/env python3
"""Görsel envanter için kombin CSV + estetik skor + kolaj üretimi."""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aesthetic_compatibility import (
    aesthetic_compatibility_score,
    color_harmony_score,
    make_aesthetic_fn,
    precompute_garment_embeddings,
)
from inventory_loader import load_production_garments as load_garments
from build_combo_collage import build_ab_collage, build_combo_collage
from stylepops_core import (
    apparent_temperature,
    generate_layered_candidates,
    interpolate_hedef_clo,
    is_valid_outfit_combo,
    score_combination,
)

VISUAL = ROOT / "data" / "visual"
LOOKUPS = ROOT / "data" / "lookups"


def load_lookups() -> tuple[dict, dict, list, dict]:
    with (LOOKUPS / "fabric_properties.json").open(encoding="utf-8") as f:
        fab = json.load(f)
    with (LOOKUPS / "coverage_ratios.json").open(encoding="utf-8") as f:
        cov = json.load(f)
    with (LOOKUPS / "target_clo_points.json").open(encoding="utf-8") as f:
        tgt = json.load(f)
    return (
        fab["thermal_categories"],
        cov["coverage_by_subcategory"],
        tgt["target_clo_points"],
        tgt["weather_scenarios"],
    )


def season_for_scenario(scenario_id: str) -> str:
    return {
        "kis_soguk_ruzgarli": "kis",
        "kis_orta": "kis",
        "sonbahar_serin": "sonbahar",
        "sonbahar_yagmurlu": "sonbahar",
        "ilkbahar_ilik": "ilkbahar",
        "ilkbahar_ruzgarli": "ilkbahar",
        "yaz_sicak": "yaz",
        "yaz_nemli": "yaz",
    }.get(scenario_id, "ilkbahar")


GENDER_SHARE = {"women": 0.62, "men": 0.38}


def select_diverse(
    scored_list: list[dict],
    quota: int,
    usage: "Counter",
    *,
    reuse_penalty: float = 0.45,
) -> list[dict]:
    """Sadece en yüksek rank'ı seçmek aynı 'en iyi' parçaları tekrar tekrar
    kullanır → 'hep aynı ürünleri görüyoruz'. Açgözlü çeşitlilik seçimi: her
    adımda rank'tan, halihazırda kullanılan parçaların tekrar cezasını düşerek
    en iyiyi seç; böylece kombinler farklı parçalara yayılır."""
    chosen: list[dict] = []
    pool = scored_list[:]
    while pool and len(chosen) < quota:
        best_i, best_adj = 0, None
        for i, r in enumerate(pool):
            pen = reuse_penalty * sum(usage[p] for p in r["piece_ids"])
            adj = float(r["rank"]) - pen
            if best_adj is None or adj > best_adj:
                best_adj, best_i = adj, i
        r = pool.pop(best_i)
        chosen.append(r)
        for p in r["piece_ids"]:
            usage[p] += 1
    return chosen


def _combo_gender_label(piece_ids: list[str], garments: dict) -> str:
    from garment_gender import combo_gender
    g = combo_gender([garments[p].get("gender", "women") for p in piece_ids if p in garments])
    return g or "women"


def generate_combinations(
    per_scenario: int = 40,
    seed: int = 42,
    *,
    fast: bool = False,
    n_candidates: int = 800,
) -> list[dict]:
    garments = load_garments()
    thermal_cats, coverage, clo_points, scenarios = load_lookups()

    from aesthetic_compatibility import has_emb_cache

    use_clip = not fast and has_emb_cache()
    if fast:
        print("Hızlı mod: FashionCLIP atlanıyor (renk + termal sıralama)")
    elif use_clip:
        print("FashionCLIP embedding önbelleği kullanılıyor (model yüklenmez)")
    else:
        print("Önbellek yok — FashionCLIP modeli yükleniyor (yavaş olabilir)…")
        precompute_garment_embeddings(garments)

    def score_aes(piece_ids: list[str]) -> dict:
        if fast:
            color = color_harmony_score(piece_ids, garments)
            return {
                "aesthetic_score": round(color, 3),
                "fashionclip_score": None,
                "color_score": round(color, 3),
                "scorer": "color_fast",
            }
        return aesthetic_compatibility_score(piece_ids, garments)

    rows: list[dict] = []
    combo_idx = 1
    # Parça kullanım sayacı cinsiyet bazında global tutulur → aynı parça tüm
    # kombinlerde tekrar etmesin (çeşitlilik). Mevsimler arası da paylaşılır.
    usage_by_gender: dict[str, Counter] = {g: Counter() for g in GENDER_SHARE}

    for i, (scenario_id, scenario) in enumerate(scenarios.items(), 1):
        season = season_for_scenario(scenario_id)
        T_app = apparent_temperature(scenario["T_hava"], scenario["RH_nem"], scenario["V_ruzgar"])
        hedef = interpolate_hedef_clo(T_app, clo_points)
        V = scenario["V_ruzgar"]
        print(f"[{i}/{len(scenarios)}] {scenario_id} ({season})")

        for gender, share in GENDER_SHARE.items():
            quota = max(1, round(per_scenario * share))
            candidates = generate_layered_candidates(
                garments, n_candidates=n_candidates, season=season,
                hedef_clo=hedef, V_ruzgar=V, seed=seed + hash(gender) % 1000,
                gender=gender,
            )
            print(f"  [{gender}] {len(candidates)} aday skorlanıyor…")
            seen: set[tuple[str, ...]] = set()
            scored_list = []
            for piece_ids in candidates:
                key = tuple(sorted(piece_ids))
                if key in seen:
                    continue
                seen.add(key)
                aes = score_aes(piece_ids)
                result = score_combination(
                    piece_ids, garments, hedef, V, thermal_cats, coverage,
                    lambda _ids, _a=aes: float(_a["aesthetic_score"]),
                    season=season,
                )
                result.update(aes)
                result["scenario_id"] = scenario_id
                result["season"] = season
                result["T_hissedilen"] = T_app
                result["hedef_Clo"] = hedef
                result["V_ruzgar"] = V
                result["gender"] = _combo_gender_label(piece_ids, garments)
                scored_list.append(result)

            scored_list.sort(key=lambda r: r["rank"], reverse=True)
            selected = select_diverse(scored_list, quota, usage_by_gender[gender])
            print(f"    {len(selected)}/{quota} [{gender}] çeşitlilikle seçildi "
                  f"(aday havuzu {len(scored_list)})")
            for r in selected:
                cid = f"VC{combo_idx:04d}"
                combo_idx += 1
                rows.append({
                    "combo_id": cid,
                    "scenario_id": scenario_id,
                    "season": season,
                    "gender": r["gender"],
                    "T_hissedilen": r["T_hissedilen"],
                    "hedef_Clo": r["hedef_Clo"],
                    "V_ruzgar": r["V_ruzgar"],
                    "piece_ids": "|".join(r["piece_ids"]),
                    "layer_count": len(r["piece_ids"]),
                    "total_Clo_C": r["total_Clo_C"],
                    "delta_Clo": r["delta_Clo"],
                    "rank": r["rank"],
                    "aesthetic_score": r["aesthetic_score"],
                    "fashionclip_score": r.get("fashionclip_score", ""),
                    "color_score": r.get("color_score", ""),
                    "scorer": r.get("scorer", ""),
                    "collage_path": f"data/assets/combos/{cid}.jpg",
                    "preference_score": "",
                    "labeler": "",
                })
    return rows


def build_collages(rows: list[dict], top_per_scenario: int = 5) -> int:
    garments = load_garments()
    built = 0
    by_scenario: dict[str, list[dict]] = {}
    for r in rows:
        by_scenario.setdefault(r["scenario_id"], []).append(r)
    for scenario_rows in by_scenario.values():
        scenario_rows.sort(key=lambda x: float(x["rank"]), reverse=True)
        for r in scenario_rows[:top_per_scenario]:
            piece_ids = [p for p in r["piece_ids"].split("|") if p]
            if build_combo_collage(piece_ids, garments, ROOT / r["collage_path"], title=r["combo_id"]):
                built += 1
    return built


def build_ab_pairs(rows: list[dict], n_pairs: int = 30, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    garments = load_garments()
    rows = [
        r for r in rows
        if is_valid_outfit_combo(
            [p for p in r["piece_ids"].split("|") if p],
            garments,
            r.get("season"),
            float(r.get("hedef_Clo", 0.9)),
        )
    ]
    pairs = []
    by_scenario: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r["scenario_id"], r.get("gender", "women"))
        by_scenario.setdefault(key, []).append(r)

    pair_idx = 1
    local_limit = max(3, n_pairs // max(len(by_scenario), 1))
    for (scenario_id, _gender), scenario_rows in by_scenario.items():
        pool = scenario_rows[:]
        rng.shuffle(pool)
        count = 0
        for i in range(0, len(pool) - 1, 2):
            if len(pairs) >= n_pairs or count >= local_limit:
                break
            a, b = pool[i], pool[i + 1]
            pid = f"AB{pair_idx:04d}"
            collage = f"data/assets/combos/{pid}.jpg"
            piece_a = [p for p in a["piece_ids"].split("|") if p]
            piece_b = [p for p in b["piece_ids"].split("|") if p]
            build_ab_collage(piece_a, piece_b, garments, ROOT / collage)
            pairs.append({
                "pair_id": pid,
                "scenario_id": scenario_id,
                "gender": _gender,
                "combo_a_id": a["combo_id"],
                "combo_b_id": b["combo_id"],
                "collage_path": collage,
                "preference_winner": "",
                "rater_id": "",
                "notes": "",
            })
            pair_idx += 1
            count += 1
    return pairs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--per-scenario", type=int, default=80)
    parser.add_argument("--ab-pairs", type=int, default=200)
    parser.add_argument("--collages", type=int, default=5)
    parser.add_argument(
        "--fast",
        action="store_true",
        help="FashionCLIP atla (~2 dk Mac). A/B pilot için yeterli.",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=None,
        help="Senaryo başına aday sayısı (varsayılan: fast=300, tam=800)",
    )
    args = parser.parse_args()
    n_candidates = args.n_candidates or (400 if args.fast else 600)

    from validate_production_wardrobe import main as validate_main
    if validate_main() != 0:
        raise SystemExit(1)

    VISUAL.mkdir(parents=True, exist_ok=True)
    rows = generate_combinations(
        per_scenario=args.per_scenario,
        fast=args.fast,
        n_candidates=n_candidates,
    )
    out_csv = VISUAL / "combinations_visual.csv"
    if rows:
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Kombin CSV: {len(rows)} satır → {out_csv}")

    n_coll = build_collages(rows, top_per_scenario=args.collages)
    print(f"Kolaj: {n_coll} görsel")

    pairs = build_ab_pairs(rows, n_pairs=args.ab_pairs)
    ab_path = VISUAL / "ab_pairs.csv"
    if pairs:
        with ab_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
            w.writeheader()
            w.writerows(pairs)
    print(f"A/B çiftleri: {len(pairs)} → {ab_path}")


if __name__ == "__main__":
    main()
