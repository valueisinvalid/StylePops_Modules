#!/usr/bin/env python3
"""Üretim envanteri ve kombin üretimi için parça uygunluk kuralları."""

from __future__ import annotations

NON_STREET_WEAR_KEYWORDS = (
    "pajama", "pyjama", "nightgown", "sleepwear", "nightwear", "shapewear",
    "loungewear", "lounge set", "lounge nightgown", "blanket hoodie",
    "corset", "waist trainer", "lingerie", "bralette", "underwire",
    "keychain", "coin purse", "phone case", "hair clip", "hairpin",
    "sunglass", "eyewear", "glasses", "wallet chain",
)

BEACH_SWIM_KEYWORDS = (
    "bikini", "swim", "pareo", "paréo", "cover-up", "cover up", "beach cover",
    "swimsuit", "swim dress", "swim short", "halter neck bikini", "kimono beach",
    "beach kimono", "one-piece swim", "swim set", "triki", "beach wear",
)

NON_WEARABLE_ACCESSORY_KEYWORDS = (
    "handbag", "hand bag", "shoulder bag", " tote", "clutch", " crossbody",
    " satchel", "backpack", " wig", "purse", "wallet", "belt buckle",
)


def text_blob(garment: dict) -> str:
    return f"{garment.get('name', '')} {garment.get('description', '')}".lower()


def exclusion_reason(garment: dict) -> str | None:
    """Sokak kombini / üretim gardırobu dışı bırakma nedeni."""
    if garment.get("active") is False:
        return "inactive"
    blob = text_blob(garment)
    for kw in NON_STREET_WEAR_KEYWORDS:
        if kw in blob:
            return f"non_street:{kw}"
    for kw in BEACH_SWIM_KEYWORDS:
        if kw in blob:
            return f"beach_swim:{kw}"
    for kw in NON_WEARABLE_ACCESSORY_KEYWORDS:
        if kw in blob:
            return f"accessory_junk:{kw}"
    if garment.get("layer_role") == "accessory" and garment.get("subcategory") == "scarf":
        if any(k in blob for k in ("sunglass", "keychain", "wallet", "bag", "glasses")):
            return "mislabeled_accessory"
    if garment.get("layer_role") == "dress" and any(
        k in blob for k in ("sandal", "heel", "shoe", "boot", "sneaker")
    ):
        return "footwear_as_dress"
    rel = garment.get("image_path", "")
    if not rel:
        return "no_image"
    return None


def is_catalog_eligible(garment: dict, *, require_image: bool = True) -> bool:
    """Üretim gardırobunda tutulabilir mi?"""
    if exclusion_reason(garment) is not None:
        return False
    if require_image:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        if not (root / garment.get("image_path", "")).exists():
            return False
    return True
