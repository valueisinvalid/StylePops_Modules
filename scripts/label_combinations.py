#!/usr/bin/env python3
"""combinations_200.csv için bootstrap etiket üretici."""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

LABELER = "StylePops Bootstrap v1.1"


def load_garments() -> dict:
    data = json.load(open(ROOT / "data/bootstrap/garments_200.json", encoding="utf-8"))
    return {g["id"]: g for g in data["garments"]}


def color_harmony_score(piece_ids: list[str], garments: dict) -> float:
    labs = [
        [garments[pid]["color_lab"]["L"], garments[pid]["color_lab"]["a"], garments[pid]["color_lab"]["b"]]
        for pid in piece_ids if pid in garments
    ]
    if len(labs) < 2:
        return 3.5
    dims = len(labs[0])
    stds = []
    for d in range(dims):
        vals = [lab[d] for lab in labs]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        stds.append(var ** 0.5)
    std = sum(stds) / len(stds)
    return max(1.0, min(5.0, 5.0 - std / 7.0))


def aesthetic_label(row: dict, eval_row: dict | None, garments: dict) -> tuple[float, str]:
    piece_ids = [p for p in str(row["piece_ids"]).split("|") if p]
    harmony = color_harmony_score(piece_ids, garments)

    if eval_row is not None:
        model_est = float(eval_row["skor_estetik"])
        base = 0.55 * model_est + 0.45 * harmony
    else:
        base = harmony

    layer_count = int(row["layer_count"])
    if layer_count >= 6:
        base += 0.3
    elif layer_count >= 4:
        base += 0.1
    elif layer_count <= 2:
        base -= 0.8

    if not str(row.get("bottom_id", "")).strip() and "dress" not in str(row.get("combo_type", "")):
        base -= 0.5

    score = max(1.0, min(5.0, round(base * 2) / 2))
    note = []
    if layer_count <= 2:
        note.append("eksik katman")
    if score >= 4:
        note.append("iyi renk uyumu")
    return score, "; ".join(note) if note else "otomatik etiket"


def thermal_label(row: dict, garments: dict) -> tuple[int, str]:
    delta = float(row["delta_Clo"])
    hedef = float(row["hedef_Clo"])
    total = float(row["total_Clo_C"])
    layer_count = int(row["layer_count"])
    has_outer = bool(str(row.get("outer_id", "")).strip())
    has_base = bool(str(row.get("base_id", "")).strip())
    has_mid = bool(str(row.get("mid_id", "")).strip())
    V = float(row["V_ruzgar"])
    rel = delta / hedef if hedef > 0 else 1.0

    piece_ids = [p for p in str(row["piece_ids"]).split("|") if p]
    subs = {garments[pid]["subcategory"] for pid in piece_ids if pid in garments}

    notes = []

    if hedef >= 1.2:  # kış
        if delta <= 0.15 or rel <= 0.10:
            score = 3
            notes.append("hedef Clo'ya çok yakın")
        elif delta <= 0.40 and has_outer and layer_count >= 5:
            score = 2
            notes.append("katmanlı ama hafif eksik")
        elif delta <= 0.55 and has_outer and (has_base or has_mid):
            score = 2
            notes.append("kabul edilebilir kış kombini")
        else:
            score = 1
            notes.append("soğuk hava için yetersiz izolasyon")
        if not has_outer and hedef > 1.4:
            score = 1
            notes.append("dış katman yok")
        if layer_count < 4 and hedef > 1.5:
            score = 1
    elif hedef >= 0.55:  # sonbahar/ilkbahar
        if delta <= 0.08:
            score = 3
        elif delta <= 0.20:
            score = 2
        else:
            score = 1
        if layer_count <= 2:
            score = min(score, 2)
            notes.append("az parça")
        if V >= 15 and "skirt" in subs and "tights" not in subs and "leggings" not in subs:
            score = min(score, 2)
            notes.append("rüzgarlı günde etek altı eksik")
    else:  # yaz
        if delta <= 0.06 and total <= 0.55:
            score = 3
            notes.append("yaz için ideal hafiflik")
        elif delta <= 0.18 and total <= 0.75:
            score = 2
        else:
            score = 1
            notes.append("yaz için fazla kalın")
        if total > 0.85:
            score = 1

    return score, "; ".join(notes) if notes else "otomatik termal etiket"


def main() -> None:
    combos_path = ROOT / "data/bootstrap/combinations_200.csv"
    eval_path = Path("/Users/o_7/Downloads/bootstrap_eval_results.csv")

    with combos_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        combos = list(reader)
        fieldnames = reader.fieldnames or []

    eval_map: dict = {}
    if eval_path.exists():
        with eval_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                eval_map[r["combo_id"]] = r

    garments = load_garments()
    aes_scores: list[float] = []
    thm_scores: list[int] = []

    for r in combos:
        ev = eval_map.get(r["combo_id"])
        aes, aes_note = aesthetic_label(r, ev, garments)
        thm, thm_note = thermal_label(r, garments)
        r["aesthetic_score"] = aes
        r["thermal_score"] = thm
        r["labeler"] = LABELER
        r["notes"] = f"estetik: {aes_note} | termal: {thm_note}"
        aes_scores.append(aes)
        thm_scores.append(thm)

    with combos_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combos)

    print(f"Labeled {len(combos)} combinations -> {combos_path}")
    print("Aesthetic:", dict(sorted(Counter(aes_scores).items())))
    print("Thermal:", dict(sorted(Counter(thm_scores).items())))
    for r in combos[:8]:
        print(f"  {r['combo_id']} aes={r['aesthetic_score']} thm={r['thermal_score']} Δ={r['delta_Clo']} layers={r['layer_count']}")


if __name__ == "__main__":
    main()
