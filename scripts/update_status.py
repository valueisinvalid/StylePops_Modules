#!/usr/bin/env python3
"""Pipeline sonrası status.json günceller."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "data" / "visual"
sys_path = ROOT / "scripts"


def main() -> None:
    import sys
    sys.path.insert(0, str(sys_path))
    from inventory_loader import catalog_meta

    status = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "production": catalog_meta("production"),
        "training_44k": catalog_meta("training"),
        "combinations": 0,
        "ab_pairs": 0,
        "preferences_logged": 0,
        "fashionclip_lora": (ROOT / "outputs" / "fashionclip_lora").exists(),
        "compatibility_head": (ROOT / "outputs" / "compatibility_head_v1.joblib").exists(),
    }

    assets_liv = ROOT / "data" / "assets" / "garments"
    assets_fp = ROOT / "data" / "assets" / "fashion_product"
    status["images_livostyle"] = len(list(assets_liv.glob("*.jpg"))) if assets_liv.exists() else 0
    status["images_fashion_product"] = len(list(assets_fp.glob("*.jpg"))) if assets_fp.exists() else 0

    for key, fname in (("combinations", "combinations_visual.csv"), ("ab_pairs", "ab_pairs.csv")):
        p = VISUAL / fname
        if p.exists():
            status[key] = max(0, sum(1 for _ in p.read_text().splitlines()) - 1)

    ppath = VISUAL / "preferences_log.csv"
    if ppath.exists():
        status["preferences_logged"] = max(0, sum(1 for _ in ppath.read_text().splitlines()) - 1)

    out = VISUAL / "status.json"
    out.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
