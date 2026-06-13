#!/usr/bin/env python3
"""Colab kombin üretimi için veri paketi (gardırop + görseller + LoRA)."""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "colab_combo_bundle.zip"
BUNDLE_ROOT = "stylepops_colab"


def main() -> None:
    prod_path = ROOT / "data" / "visual" / "garments_production.json"
    if not prod_path.exists():
        raise SystemExit("Önce: python scripts/curate_production_wardrobe.py --supplement 100")

    garments = json.loads(prod_path.read_text(encoding="utf-8"))["garments"]
    staging = ROOT / "outputs" / "_colab_staging" / BUNDLE_ROOT
    if staging.exists():
        shutil.rmtree(staging.parent)
    staging.mkdir(parents=True)

    for rel in (
        "data/visual/garments_production.json",
        "data/visual/inventory_registry.json",
    ):
        src = ROOT / rel
        dst = staging / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    lookups = ROOT / "data" / "lookups"
    shutil.copytree(lookups, staging / "data" / "lookups")

    copied = 0
    missing = []
    for g in garments:
        rel = g.get("image_path", "")
        if not rel:
            missing.append(g["id"])
            continue
        src = ROOT / rel
        dst = staging / rel
        if not src.exists():
            missing.append(g["id"])
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    lora = ROOT / "outputs" / "fashionclip_lora"
    if lora.exists() and (lora / "adapter_config.json").exists():
        shutil.copytree(lora, staging / "outputs" / "fashionclip_lora")
        print(f"LoRA dahil: {lora}")
    else:
        print("Uyarı: LoRA yok — Colab base FashionCLIP kullanır")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging.parent))

    shutil.rmtree(staging.parent)
    mb = OUT.stat().st_size / (1024 * 1024)
    print(f"Paket: {OUT} ({mb:.1f} MB)")
    print(f"Görsel: {copied}/{len(garments)}")
    if missing:
        print(f"Eksik görsel: {len(missing)} parça")
    print()
    print("Drive'a yükle:")
    print("  MyDrive/StylePops_colab/colab_combo_bundle.zip")
    print("Colab notebook: notebooks/StylePops_Generate_Combinations.ipynb")


if __name__ == "__main__":
    main()
