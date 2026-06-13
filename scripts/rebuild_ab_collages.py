#!/usr/bin/env python3
"""Mevcut ab_pairs.csv için A/B kolajlarını yeniden üretir (hızlı, CLIP yok)."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_combo_collage import build_ab_collage
from inventory_loader import load_production_garments

VISUAL = ROOT / "data" / "visual"


def main() -> None:
    pairs_path = VISUAL / "ab_pairs.csv"
    combos_path = VISUAL / "combinations_visual.csv"
    if not pairs_path.exists():
        raise SystemExit(f"Yok: {pairs_path}")

    garments = load_production_garments()
    combos = {r["combo_id"]: r for r in csv.DictReader(combos_path.open(encoding="utf-8"))}

    pairs = list(csv.DictReader(pairs_path.open(encoding="utf-8")))
    built = 0
    for p in pairs:
        a = combos.get(p["combo_a_id"])
        b = combos.get(p["combo_b_id"])
        if not a or not b:
            continue
        ids_a = [x for x in a["piece_ids"].split("|") if x]
        ids_b = [x for x in b["piece_ids"].split("|") if x]
        out = ROOT / p["collage_path"]
        if build_ab_collage(ids_a, ids_b, garments, out):
            built += 1
        if built % 25 == 0:
            print(f"  … {built}/{len(pairs)}")

    print(f"A/B kolaj yenilendi: {built}/{len(pairs)}")


if __name__ == "__main__":
    main()
