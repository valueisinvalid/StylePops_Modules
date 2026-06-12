#!/usr/bin/env python3
"""Görsel envanter yükleme — registry tabanlı tek giriş noktası."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VISUAL = ROOT / "data" / "visual"
REGISTRY_PATH = VISUAL / "inventory_registry.json"


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {
            "production_wardrobe": "garments_livostyle.json",
            "training_corpus": "garments_fashion_product.json",
        }
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def _load_catalog(filename: str) -> dict[str, dict]:
    path = VISUAL / filename
    if not path.exists():
        # Geriye uyumluluk
        legacy = VISUAL / "garments_300.json"
        if legacy.exists() and filename.startswith("garments_livostyle"):
            path = legacy
        else:
            return {}
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    garments = data.get("garments", data if isinstance(data, list) else [])
    return {g["id"]: g for g in garments}


def load_production_garments() -> dict[str, dict]:
    reg = load_registry()
    return _load_catalog(reg.get("production_wardrobe", "garments_livostyle.json"))


def load_training_garments() -> dict[str, dict]:
    reg = load_registry()
    return _load_catalog(reg.get("training_corpus", "garments_fashion_product.json"))


def load_all_garments() -> dict[str, dict]:
    out = {}
    out.update(load_training_garments())
    out.update(load_production_garments())
    return out


def catalog_meta(which: str = "production") -> dict:
    reg = load_registry()
    fname = (
        reg.get("production_wardrobe", "garments_livostyle.json")
        if which == "production"
        else reg.get("training_corpus", "garments_fashion_product.json")
    )
    path = VISUAL / fname
    if not path.exists() and which == "production":
        path = VISUAL / "garments_300.json"
    if not path.exists():
        return {"exists": False, "path": str(path)}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "count": data.get("count", len(data.get("garments", []))),
        "source": data.get("source"),
        "license": data.get("license"),
        "snapshot_date": data.get("snapshot_date"),
    }
