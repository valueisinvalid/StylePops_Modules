#!/usr/bin/env python3
"""Üretim gardırobunda giyim dışı parça kalmadığını doğrular — kombin üretiminden önce çalıştır."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from garment_eligibility import exclusion_reason, non_garment_reason
from inventory_loader import load_production_garments

MAX_REPORT = 25


def main() -> int:
    garments = load_production_garments()
    if not garments:
        print("HATA: üretim gardırobu boş", file=sys.stderr)
        return 1

    bad: list[tuple[str, str, str]] = []
    for g in garments.values():
        reason = non_garment_reason(g) or exclusion_reason(g)
        if reason and ("non_garment" in reason or reason.startswith("accessory_junk")):
            bad.append((g["id"], g.get("name", "")[:60], reason))

    if bad:
        print(f"HATA: {len(bad)} giyim-dışı parça üretim envanterinde:", file=sys.stderr)
        for item in bad[:MAX_REPORT]:
            print(f"  {item[0]} | {item[1]} | {item[2]}", file=sys.stderr)
        if len(bad) > MAX_REPORT:
            print(f"  … ve {len(bad) - MAX_REPORT} tane daha", file=sys.stderr)
        return 1

    epoch = {
        "wardrobe_count": len(garments),
        "validated": True,
        "note": "Kombin/A-B yenilemeden önce bu dosyayı güncelle; tercihler geçersiz olur.",
    }
    out = ROOT / "data" / "visual" / "data_epoch.json"
    out.write_text(json.dumps(epoch, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"OK: {len(garments)} parça — giyim dışı yok → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
