#!/usr/bin/env python3
"""Parça cinsiyet çıkarımı (kadın / erkek / unisex) ve çocuk ürünü tespiti.

- SP (Fashion Product 44K) parçalarında `fp_meta.gender` güvenilir kaynaktır.
- LV (Livostyle) parçalarında metinden çıkarım yapılır; marka kadın ağırlıklı
  olduğundan açık erkek/unisex sinyali yoksa varsayılan "women" kabul edilir.
"""

from __future__ import annotations

import re

WOMEN = "women"
MEN = "men"
UNISEX = "unisex"

_KIDS_RE = re.compile(
    r"\b(boys?|girls?|toddler|infant|infants?|kids?|children|child|junior|"
    r"newborn|nursery|çocuk|bebek|kiz\s*cocuk|erkek\s*cocuk)\b",
    re.I,
)
# "baby doll", "baby rib" gibi kadın giyim kalıpları çocuk sinyali değildir
_BABY_FASHION_RE = re.compile(r"\bbaby\s+(doll|rib|tee|blue|pink)\b", re.I)

_MEN_RE = re.compile(r"\b(men|men'?s|mens|herren|erkek|male|gentlemen|boyfriend)\b", re.I)
_WOMEN_RE = re.compile(r"\b(women|women'?s|womens|woman|ladies|lady|female|kadın|damen|girlfriend)\b", re.I)
_UNISEX_RE = re.compile(r"\bunisex\b", re.I)


def _blob(g: dict) -> str:
    return f"{g.get('name', '')} {g.get('description', '')}"


def _fp_gender(g: dict) -> str | None:
    fm = g.get("fp_meta") or {}
    val = (fm.get("gender") or "").strip().lower()
    return val or None


def is_kids_item(g: dict) -> bool:
    """Çocuk/bebek ürünü mü? (üretim gardırobundan tamamen çıkarılır)"""
    fg = _fp_gender(g)
    if fg in ("boys", "girls", "kids"):
        return True
    blob = _blob(g)
    if _BABY_FASHION_RE.search(blob):
        return False
    return bool(_KIDS_RE.search(blob))


def infer_gender(g: dict) -> str:
    """Parçanın hedef cinsiyeti: 'women' | 'men' | 'unisex'."""
    fg = _fp_gender(g)
    if fg == "men":
        return MEN
    if fg == "women":
        return WOMEN
    if fg == "unisex":
        return UNISEX

    blob = _blob(g)
    if _UNISEX_RE.search(blob):
        return UNISEX
    has_men = bool(_MEN_RE.search(blob))
    has_women = bool(_WOMEN_RE.search(blob))
    if has_men and not has_women:
        return MEN
    if has_women and not has_men:
        return WOMEN
    if has_men and has_women:
        return UNISEX
    # Livostyle açık sinyal yok → marka kadın ağırlıklı
    return WOMEN


def genders_compatible(a: str, b: str) -> bool:
    """İki parça aynı kombinde olabilir mi? Unisex her ikisiyle uyumlu."""
    if a == UNISEX or b == UNISEX:
        return True
    return a == b


def combo_gender(genders: list[str]) -> str | None:
    """Kombinin baskın cinsiyeti; karışık (men+women) ise None."""
    concrete = {g for g in genders if g in (MEN, WOMEN)}
    if len(concrete) > 1:
        return None
    if concrete:
        return next(iter(concrete))
    return UNISEX
