"""StylePops termal model ve katmanlı kombin mantığı — İP-2 / İP-3 çekirdeği."""

from __future__ import annotations

import random
from typing import Any

# Katman slotları: aynı slotta tek parça, farklı slotlar üst üste giyilebilir
LAYER_SLOTS = ("base", "mid", "outer", "bottom", "dress", "footwear", "accessory")

SUBCATEGORY_TO_SLOT = {
    "tshirt": "base", "tank_top": "base", "thermal_base": "base",
    "blouse": "mid", "shirt": "mid", "sweater": "mid", "hoodie": "mid", "cardigan": "mid",
    "blazer": "outer", "jacket": "outer", "coat": "outer",
    "padded_coat": "outer", "raincoat": "outer",
    "jeans": "bottom", "trousers": "bottom", "chinos": "bottom",
    "shorts": "bottom", "skirt": "bottom", "leggings": "bottom",
    "dress_short": "dress", "dress_midi": "dress", "dress_long": "dress",
    "boots": "footwear", "sneakers": "footwear", "sandals": "footwear",
    "scarf": "accessory", "hat": "accessory", "tights": "accessory",
}


def garment_slot(garment: dict) -> str:
    return garment.get("layer_role") or SUBCATEGORY_TO_SLOT.get(
        garment["subcategory"], garment["category"]
    )


def effective_clo(garment: dict, thermal_cats: dict, coverage_defaults: dict) -> float:
    """
    Giysi düzeyinde Clo (thermal_category) + kaplama oranı normalizasyonu.

    thermal_category değerleri parça bazlı tipik Clo'yu temsil eder (ASHRAE ölçeği).
    Kısa etek vs uzun etek / pantolon farkı coverage_ratio ile ölçeklenir.
    """
    base = thermal_cats[garment["thermal_category"]]["clo"]
    default_cov = coverage_defaults.get(garment["subcategory"], 0.70)
    actual_cov = garment["coverage_ratio"]
    if default_cov <= 0:
        return base
    return round(base * (actual_cov / default_cov), 4)


def effective_ret(garment: dict, thermal_cats: dict, coverage_defaults: dict) -> float:
    base = thermal_cats[garment["thermal_category"]]["ret"]
    default_cov = coverage_defaults.get(garment["subcategory"], 0.70)
    actual_cov = garment["coverage_ratio"]
    breathability_scale = 1.0 + 0.25 * max(0.0, (default_cov - actual_cov) / default_cov)
    return round(base * breathability_scale, 4)


def ensemble_total_clo(pieces: list[dict], thermal_cats: dict, coverage_defaults: dict) -> float:
    """ISO 9920 bootstrap: kombinasyon Clo = parça Clo'larının toplamı."""
    return round(
        sum(effective_clo(p, thermal_cats, coverage_defaults) for p in pieces), 4
    )


def wind_chill_c(T_c: float, V_kmh: float) -> float:
    if T_c > 10 or V_kmh < 4.8:
        return T_c
    v_ms = V_kmh / 3.6
    return 13.12 + 0.6215 * T_c - 11.37 * (v_ms ** 0.16) + 0.3965 * T_c * (v_ms ** 0.16)


def heat_index_c(T_c: float, RH: float) -> float:
    if T_c < 27:
        return T_c
    Tf = T_c * 9 / 5 + 32
    HI = (
        -42.379 + 2.04901523 * Tf + 10.14333127 * RH
        - 0.22475541 * Tf * RH - 0.00683783 * Tf ** 2
        - 0.05481717 * RH ** 2 + 0.00122874 * Tf ** 2 * RH
        + 0.00085282 * Tf * RH ** 2 - 0.00000199 * Tf ** 2 * RH ** 2
    )
    return (HI - 32) * 5 / 9


def apparent_temperature(T_hava: float, RH: float, V_ruzgar: float) -> float:
    if T_hava <= 10 and V_ruzgar >= 5:
        return round(wind_chill_c(T_hava, V_ruzgar), 2)
    if T_hava >= 27:
        return round(heat_index_c(T_hava, RH), 2)
    return round(T_hava, 2)


def interpolate_hedef_clo(T: float, clo_points: list[dict]) -> float:
    pts = sorted(clo_points, key=lambda p: p["T_celsius"])
    if T <= pts[0]["T_celsius"]:
        return pts[0]["clo"]
    if T >= pts[-1]["T_celsius"]:
        return pts[-1]["clo"]
    for i in range(len(pts) - 1):
        t1, c1 = pts[i]["T_celsius"], pts[i]["clo"]
        t2, c2 = pts[i + 1]["T_celsius"], pts[i + 1]["clo"]
        if t1 <= T <= t2:
            return round(c1 + (T - t1) * (c2 - c1) / (t2 - t1), 4)
    return pts[-1]["clo"]


def garment_fits_season(g: dict, season: str | None, hedef_clo: float) -> bool:
    if not season:
        return True
    if season in g.get("season_usable", [season]):
        return True
    slot = garment_slot(g)
    if slot in ("footwear", "accessory"):
        return False
    if is_cold_context(season, hedef_clo):
        if g.get("subcategory") in ("shorts", "dress_short", "tank_top"):
            return False
        return slot in ("base", "mid", "outer", "bottom", "dress")
    if is_warm_context(season, hedef_clo):
        if g.get("subcategory") in ("padded_coat", "coat"):
            return False
    return False


def inventory_by_slot(
    garments: dict[str, dict],
    season: str | None = None,
    hedef_clo: float = 0.6,
) -> dict[str, list[str]]:
    """Envanteri katman slotlarına göre grupla."""
    slots: dict[str, list[str]] = {s: [] for s in LAYER_SLOTS}
    for gid, g in garments.items():
        if not garment_fits_season(g, season, hedef_clo):
            continue
        slot = garment_slot(g)
        if slot in slots:
            slots[slot].append(gid)
    return slots


WEARABLE_ACCESSORY_SUBS = frozenset({"scarf", "hat", "tights"})
JEWELRY_KEYWORDS = (
    "ring", "earring", "necklace", "bracelet", "moissanite", "jewelry",
    "pendant", "brooch", "anklet", "choker",
)
FOOTWEAR_KEYWORDS = (
    "sandal", "heel", "shoe", "boot", "sneaker", "pump", "mule", "loafer", "flat",
)
SET_KEYWORDS = (" set", "two-piece", "two piece", "lounge set", "co-ord", "coord set")
SUMMER_FOOTWEAR_KEYWORDS = (
    "sandal", "open toe", "flip flop", "slide", "wedge sandal", "heel sandal",
    "flat sandal", "mule", "toe loop",
)
WINTER_FOOTWEAR_KEYWORDS = (
    "boot", "ankle boot", "chelsea", "snow boot", "fur lined", "shearling", "winter boot",
)
SUMMER_ACCESSORY_KEYWORDS = ("sun hat", "sun visor", "visor", "straw hat", "beach hat", "floppy hat")
WINTER_ACCESSORY_KEYWORDS = ("beanie", "pompom", "winter hat", "knit hat", "scarf", "earmuff")
BEACH_SWIM_KEYWORDS = (
    "bikini", "swim", "pareo", "paréo", "cover-up", "cover up", "beach cover",
    "swimsuit", "swim dress", "swim short", "halter neck bikini", "kimono beach",
    "beach kimono", "one-piece swim", "swim set", "triki",
)
NON_OUTFIT_KEYWORDS = (
    "keychain", "coin purse", "sunglass", "eyewear", "glasses", "phone case",
    "hair clip", "hairpin", "wallet chain",
    "pajama", "pyjama", "nightgown", "shapewear", "loungewear", "sleepwear",
    "panty", "panties", "thong", "briefs", "underwear", "undergarment",
    "shaping panty", "shaping short", "control brief", "intimates",
    "sports bra", "bra top", "corset", "lingerie",
)
UNDERWEAR_BOTTOM_KEYWORDS = (
    "panty", "panties", "thong", "brief", "underwear", "shaping", "intimates",
    "control top", "girdle",
)


def _text_blob(garment: dict) -> str:
    return f"{garment.get('name', '')} {garment.get('description', '')}".lower()


def is_beach_swim_garment(garment: dict) -> bool:
    blob = _text_blob(garment)
    return any(k in blob for k in BEACH_SWIM_KEYWORDS)


def is_everyday_dress(garment: dict) -> bool:
    return is_dress_piece(garment) and not is_beach_swim_garment(garment)


HEAVY_OUTER_KEYWORDS = (
    "parka", "puffer", "down jacket", "down coat", "wool coat", "woolen coat",
    "overcoat", "shearling", "teddy coat", "teddy bear coat", "quilted",
    "padded coat", "padded jacket", "fur coat", "peacoat", "pea coat",
    "longline coat", "long wool", "winter coat", "duffle coat", "sherpa",
    "puffer coat", "puffer jacket",
)
MID_OUTER_KEYWORDS = ("fleece coat", "wool blend", "wool-blend", "trench")
LIGHT_OUTER_KEYWORDS = (
    "shacket", "fleece shacket", "overshirt", "shirt jacket", "flannel shirt",
    "denim jacket", "jean jacket", "cropped denim", "blazer", "moto", "rider",
    "leather jacket", "bomber", "windbreaker", "utility jacket", "vest",
)


def outer_warmth_tier(garment: dict) -> str:
    """Dış katman sıcaklık sınıfı.

    LV verisinde blazer/kot/bomber çoğu zaman subcategory='coat' olarak gelir;
    bu yüzden subcategory'e değil metindeki anahtar kelimelere güveniyoruz."""
    sub = garment.get("subcategory", "")
    blob = _text_blob(garment)
    if any(k in blob for k in HEAVY_OUTER_KEYWORDS):
        return "heavy"
    if "trench" in blob and "coat" in blob:
        return "heavy"
    if sub == "padded_coat":
        return "heavy"
    if any(k in blob for k in LIGHT_OUTER_KEYWORDS):
        return "light"
    if any(k in blob for k in MID_OUTER_KEYWORDS):
        return "mid"
    if sub == "raincoat":
        return "mid"
    # 'coat' sözcüğü güçlü kış sinyali olmadan → orta
    if sub == "coat" or "coat" in blob:
        return "mid"
    if sub in ("blazer", "jacket"):
        return "light"
    return "light"


_SPORT_RE = None
_FORMAL_RE = None


def garment_style(garment: dict) -> str:
    """Parça stili: 'sport' | 'formal' | 'casual' (kombin tutarlılığı için)."""
    import re
    global _SPORT_RE, _FORMAL_RE
    if _SPORT_RE is None:
        _SPORT_RE = re.compile(
            r"\b(active(wear)?|sport|gym|workout|yoga|athletic|athleisure|jogger|"
            r"track pant|track suit|legging|running|tennis|football|basketball|"
            r"sports bra|sweatpant)\b", re.I)
        _FORMAL_RE = re.compile(
            r"\b(blazer|formal|suit|tailored|dress pant|dress trouser|oxford shirt|"
            r"tuxedo|sport coat|pleated dress pant)\b", re.I)
    meta = garment.get("fp_meta") or {}
    usage = (meta.get("usage") or "").strip().lower()
    blob = _text_blob(garment)
    if usage == "sports" or _SPORT_RE.search(blob):
        return "sport"
    if usage == "formal" or _FORMAL_RE.search(blob):
        return "formal"
    return "casual"


def is_outfit_eligible(garment: dict, season: str | None, hedef_clo: float) -> bool:
    if not is_wearable_in_combos(garment):
        return False
    if is_beach_swim_garment(garment):
        return False
    if any(k in _text_blob(garment) for k in NON_OUTFIT_KEYWORDS):
        return False
    return is_season_appropriate_garment(garment, garment_slot(garment), season, hedef_clo)


def is_cold_context(season: str | None, hedef_clo: float) -> bool:
    if hedef_clo >= 1.0:
        return True
    return season in ("kis", "sonbahar") and hedef_clo >= 0.8


def is_warm_context(season: str | None, hedef_clo: float) -> bool:
    if hedef_clo < 0.45:
        return True
    return season == "yaz"


def footwear_season_tier(garment: dict) -> str:
    sub = garment.get("subcategory", "")
    blob = _text_blob(garment)
    if sub == "boots" or any(k in blob for k in WINTER_FOOTWEAR_KEYWORDS):
        return "winter"
    if sub == "sandals" or any(k in blob for k in SUMMER_FOOTWEAR_KEYWORDS):
        return "summer"
    return "neutral"


def is_season_appropriate_footwear(
    garment: dict, season: str | None, hedef_clo: float,
) -> bool:
    tier = footwear_season_tier(garment)
    if is_cold_context(season, hedef_clo):
        return tier != "summer"
    if is_warm_context(season, hedef_clo):
        return tier != "winter"
    return True


def is_season_appropriate_accessory(
    garment: dict, season: str | None, hedef_clo: float,
) -> bool:
    if not is_thermal_accessory(garment):
        return False
    blob = _text_blob(garment)
    if is_cold_context(season, hedef_clo):
        if any(k in blob for k in SUMMER_ACCESSORY_KEYWORDS):
            return False
        if hedef_clo >= 1.2 and garment.get("subcategory") == "hat":
            if not any(
                k in blob
                for k in WINTER_ACCESSORY_KEYWORDS + ("beanie", "beret", "knit", "pompom", "wool")
            ):
                return False
        return True
    if is_warm_context(season, hedef_clo):
        if garment.get("subcategory") == "scarf" and hedef_clo < 0.5:
            return False
    return True


def is_season_appropriate_garment(
    garment: dict,
    slot: str,
    season: str | None,
    hedef_clo: float,
) -> bool:
    if is_cold_context(season, hedef_clo):
        if garment.get("subcategory") == "shorts":
            return False
        if garment.get("subcategory") == "dress_short":
            return False
        if garment.get("sleeve") == "sleeveless" and slot in ("base", "mid", "outer"):
            return False
    if is_warm_context(season, hedef_clo):
        if garment.get("subcategory") in {"padded_coat", "coat"} and hedef_clo < 0.55:
            return False
    if slot == "footwear":
        return is_season_appropriate_footwear(garment, season, hedef_clo)
    if slot == "accessory":
        return is_season_appropriate_accessory(garment, season, hedef_clo)
    return True


def is_jewelry_or_bag(garment: dict) -> bool:
    blob = _text_blob(garment)
    if any(kw in blob for kw in JEWELRY_KEYWORDS):
        return True
    if any(kw in blob for kw in NON_OUTFIT_KEYWORDS):
        return True
    if any(
        kw in blob
        for kw in (
            "handbag", "hand bag", "shoulder bag", " tote", "clutch", " crossbody",
            " satchel", " wallet", "backpack", " wig", "purse",
        )
    ):
        return True
    return False


def is_valid_bottom_piece(garment: dict) -> bool:
    if is_dress_piece(garment) or is_coordinated_set(garment):
        return False
    blob = _text_blob(garment)
    if any(k in blob for k in ("top and", "tee and", "shirt and", " set", "lounge")):
        return False
    if any(k in blob for k in UNDERWEAR_BOTTOM_KEYWORDS):
        return False
    return garment.get("subcategory") in {
        "jeans", "trousers", "chinos", "shorts", "skirt", "leggings",
    }


def looks_like_footwear(garment: dict) -> bool:
    return any(kw in _text_blob(garment) for kw in FOOTWEAR_KEYWORDS)


def is_dress_piece(garment: dict) -> bool:
    if garment.get("layer_role") == "dress":
        return True
    sub = garment.get("subcategory", "")
    if sub.startswith("dress"):
        return True
    name = garment.get("name", "").lower()
    if "dress" in name and "address" not in name and "dressing" not in name:
        return True
    if any(k in _text_blob(garment) for k in ("jumpsuit", "romper", "one-piece")):
        return True
    return False


def is_coordinated_set(garment: dict) -> bool:
    blob = _text_blob(garment)
    return any(k in blob for k in SET_KEYWORDS)


def is_wearable_in_combos(garment: dict) -> bool:
    from garment_eligibility import is_catalog_eligible

    if not is_catalog_eligible(garment):
        return False
    if garment.get("layer_role") == "dress" and looks_like_footwear(garment):
        return False
    return True


def is_thermal_accessory(garment: dict) -> bool:
    if garment.get("subcategory") not in WEARABLE_ACCESSORY_SUBS:
        return False
    if is_jewelry_or_bag(garment):
        return False
    blob = _text_blob(garment)
    sub = garment.get("subcategory")
    if sub == "scarf":
        return any(k in blob for k in ("scarf", "shawl", "wrap", "muffler"))
    if sub == "hat":
        if "keychain" in blob:
            return False
        return any(k in blob for k in ("hat", "cap", "beanie", "beret", "visor", "knit", "pompom"))
    if sub == "tights":
        return "tight" in blob or "hosiery" in blob or "stocking" in blob
    return False


def outfit_coherence_penalty(pieces: list[dict]) -> float:
    """Stil/tür uyumsuzluğu — aşırı katman ve çakışan parçalar."""
    penalty = 0.0
    dress_items = [p for p in pieces if is_dress_piece(p)]
    set_items = [p for p in pieces if is_coordinated_set(p)]
    slots = {garment_slot(p) for p in pieces}
    subs = {p["subcategory"] for p in pieces}

    if dress_items:
        if "bottom" in slots or "base" in slots:
            penalty += 3.5
        if len(dress_items) > 1:
            penalty += 2.0
        if any(garment_slot(p) == "mid" for p in pieces if not is_dress_piece(p)):
            penalty += 1.5

    if set_items:
        if "bottom" in slots and not any(is_coordinated_set(p) for p in pieces if garment_slot(p) == "bottom"):
            penalty += 2.5
        if "base" in slots and not any(is_coordinated_set(p) for p in pieces if garment_slot(p) == "base"):
            penalty += 1.5

    if subs & {"shorts"} and subs & {"leggings", "jeans", "chinos", "trousers"}:
        penalty += 2.0
    if len(pieces) > 5:
        penalty += 0.8 * (len(pieces) - 5)
    if any(is_jewelry_or_bag(p) for p in pieces):
        penalty += 4.0

    # Stil uyumu: spor ve formal aynı kombinde olmamalı
    styles = {garment_style(p) for p in pieces}
    if "sport" in styles and "formal" in styles:
        penalty += 4.0

    return penalty


def min_outfit_requirements_met(
    pieces: list[dict],
    season: str | None,
    hedef_clo: float,
) -> bool:
    if len(pieces) < 3:
        return False
    slots = {garment_slot(p) for p in pieces}
    if "footwear" not in slots:
        return False
    if is_cold_context(season, hedef_clo) and "outer" not in slots:
        return False
    if is_cold_context(season, hedef_clo):
        outers = [p for p in pieces if garment_slot(p) == "outer"]
        if hedef_clo >= 1.4:
            if not outers or not all(outer_warmth_tier(p) == "heavy" for p in outers):
                return False
        elif hedef_clo >= 1.1:
            if not outers or any(outer_warmth_tier(p) == "light" for p in outers):
                return False
    if any(is_coordinated_set(p) for p in pieces):
        if "footwear" in slots and len(pieces) >= 3:
            return True
    if any(is_everyday_dress(p) for p in pieces):
        return len(pieces) >= 3
    if "bottom" not in slots:
        return False
    if "base" not in slots and "mid" not in slots:
        return False
    return True


def season_coherence_penalty(
    pieces: list[dict],
    season: str | None,
    hedef_clo: float,
) -> float:
    penalty = 0.0
    for p in pieces:
        slot = garment_slot(p)
        if not is_season_appropriate_garment(p, slot, season, hedef_clo):
            penalty += 2.5
        if is_cold_context(season, hedef_clo) and slot == "footwear" and footwear_season_tier(p) == "summer":
            penalty += 3.0
        if (
            is_cold_context(season, hedef_clo)
            and slot == "outer"
            and outer_warmth_tier(p) == "light"
            and hedef_clo >= 1.2
        ):
            penalty += 1.5
        if is_cold_context(season, hedef_clo):
            blob = _text_blob(p)
            if any(k in blob for k in SUMMER_ACCESSORY_KEYWORDS):
                penalty += 2.0
    return penalty


def is_valid_outfit_combo(
    piece_ids: list[str],
    garments: dict[str, dict],
    season: str | None = None,
    hedef_clo: float = 0.6,
) -> bool:
    pieces = [garments[pid] for pid in piece_ids if pid in garments]
    if len(pieces) < 2:
        return False
    from garment_gender import combo_gender
    if combo_gender([p.get("gender", "women") for p in pieces]) is None:
        return False
    if any(not is_outfit_eligible(p, season, hedef_clo) for p in pieces):
        return False
    if outfit_coherence_penalty(pieces) >= 2.5:
        return False
    slots = [garment_slot(p) for p in pieces]
    if slots.count("footwear") > 1:
        return False
    dress_items = [p for p in pieces if is_dress_piece(p)]
    if dress_items and ("base" in slots or "bottom" in slots):
        return False
    if not min_outfit_requirements_met(pieces, season, hedef_clo):
        return False
    if season_coherence_penalty(pieces, season, hedef_clo) >= 4.0:
        return False
    for p in pieces:
        slot = garment_slot(p)
        if not is_season_appropriate_garment(p, slot, season, hedef_clo):
            return False
    return True


def accessory_hints(pieces: list[dict], hedef_clo: float, V_ruzgar: float) -> list[str]:
    """Eksik aksesuar önerileri (kural tabanlı)."""
    hints = []
    subs = {p["subcategory"] for p in pieces}
    has = subs

    if hedef_clo > 1.0:
        if "scarf" not in has:
            hints.append("atkı")
        if "hat" not in has:
            hints.append("şapka")
    if V_ruzgar >= 15 and "skirt" in subs and "tights" not in has and "leggings" not in has:
        hints.append("kilotlu çorap")
    if hedef_clo > 0.9 and "thermal_base" not in has and "tshirt" not in has:
        hints.append("iç katman (tişört/termal)")
    return hints


def _slot_candidates(
    slots: dict[str, list[str]],
    garments: dict[str, dict],
    slot: str,
    *,
    thermal_accessory_only: bool = False,
    dress_only: bool = False,
    everyday_dress_only: bool = False,
    set_only: bool = False,
    exclude_dress: bool = False,
    season: str | None = None,
    hedef_clo: float = 0.6,
) -> list[str]:
    ids = slots.get(slot, [])
    out = []
    for gid in ids:
        g = garments.get(gid)
        if not g or not is_outfit_eligible(g, season, hedef_clo):
            continue
        if thermal_accessory_only and not is_thermal_accessory(g):
            continue
        if everyday_dress_only and not is_everyday_dress(g):
            continue
        if dress_only and not is_dress_piece(g):
            continue
        if set_only and not is_coordinated_set(g):
            continue
        if exclude_dress and is_dress_piece(g):
            continue
        if slot == "bottom" and not is_valid_bottom_piece(g):
            continue
        out.append(gid)
    return out


def _pick_outer(
    pool: list[str],
    garments: dict[str, dict],
    rng: random.Random,
    season: str | None,
    hedef_clo: float,
) -> str | None:
    if not pool:
        return None
    if is_cold_context(season, hedef_clo):
        tiers = ("heavy",) if hedef_clo >= 1.4 else ("heavy", "mid")
        for tier in tiers:
            tiered = [g for g in pool if outer_warmth_tier(garments[g]) == tier]
            if tiered:
                return rng.choice(tiered)
        return None
    return rng.choice(pool)


def _pick_footwear(
    pool: list[str],
    garments: dict[str, dict],
    rng: random.Random,
    season: str | None,
    hedef_clo: float,
) -> str | None:
    if not pool:
        return None
    if is_cold_context(season, hedef_clo):
        winter = [g for g in pool if footwear_season_tier(garments[g]) == "winter"]
        if winter:
            return rng.choice(winter)
        neutral = [g for g in pool if footwear_season_tier(garments[g]) == "neutral"]
        if neutral:
            return rng.choice(neutral)
        return None
    if is_warm_context(season, hedef_clo):
        summer = [g for g in pool if footwear_season_tier(garments[g]) in ("summer", "neutral")]
        if summer:
            return rng.choice(summer)
    return rng.choice(pool)


def _build_combo_pools(
    slots: dict[str, list[str]],
    garments: dict[str, dict],
    season: str | None,
    hedef_clo: float,
) -> dict[str, list[str]]:
    """Kombin üretimi için filtrelenmiş slot havuzları — bir kez hesaplanır."""
    dress_pool = _slot_candidates(
        slots, garments, "dress", everyday_dress_only=True, season=season, hedef_clo=hedef_clo,
    )
    dress_pool += _slot_candidates(
        slots, garments, "mid", everyday_dress_only=True, season=season, hedef_clo=hedef_clo,
    )
    set_pool = (
        _slot_candidates(slots, garments, "base", set_only=True, season=season, hedef_clo=hedef_clo)
        + _slot_candidates(slots, garments, "dress", set_only=True, season=season, hedef_clo=hedef_clo)
    )
    set_pool = [gid for gid in set_pool if not is_beach_swim_garment(garments[gid])]
    return {
        "dress": dress_pool,
        "set": set_pool,
        "outer": _slot_candidates(slots, garments, "outer", exclude_dress=True, season=season, hedef_clo=hedef_clo),
        "base": [
            gid for gid in _slot_candidates(
                slots, garments, "base", exclude_dress=True, set_only=False, season=season, hedef_clo=hedef_clo,
            )
            if not is_coordinated_set(garments[gid])
        ],
        "mid": _slot_candidates(slots, garments, "mid", exclude_dress=True, season=season, hedef_clo=hedef_clo),
        "bottom": _slot_candidates(slots, garments, "bottom", exclude_dress=True, season=season, hedef_clo=hedef_clo),
        "footwear": _slot_candidates(slots, garments, "footwear", season=season, hedef_clo=hedef_clo),
        "accessory": _slot_candidates(
            slots, garments, "accessory", thermal_accessory_only=True, season=season, hedef_clo=hedef_clo,
        ),
    }


def build_layered_combo(
    slots: dict[str, list[str]],
    hedef_clo: float,
    V_ruzgar: float,
    rng: random.Random,
    garments: dict[str, dict] | None = None,
    season: str | None = None,
    pools: dict[str, list[str]] | None = None,
) -> list[str]:
    """
    Anlamlı kombin şablonları: elbise / set / ayrı parçalar.
    Takı ve çakışan katmanlar üretilmez.
    """
    if not garments:
        garments = {}

    def pick_from(
        pool: list[str],
        mark_slots: tuple[str, ...] = (),
    ) -> str | None:
        if not pool:
            return None
        gid = rng.choice(pool)
        return gid

    combo: list[str] = []
    used_slots: set[str] = set()

    def add(gid: str | None, *mark: str) -> None:
        if not gid or gid in combo:
            return
        combo.append(gid)
        for s in mark:
            used_slots.add(s)

    if pools is None:
        pools = _build_combo_pools(slots, garments, season, hedef_clo)

    dress_pool = pools["dress"]
    set_pool = pools["set"]
    outer_pool = pools["outer"]
    base_pool = pools["base"]
    mid_pool = pools["mid"]
    bottom_pool = pools["bottom"]
    footwear_pool = pools["footwear"]
    acc_pool = pools["accessory"]

    if is_warm_context(season, hedef_clo):
        route_choices = (["separates"] * 7) + (["set"] * 2) + (["dress"] * 1)
    elif is_cold_context(season, hedef_clo):
        route_choices = (["separates"] * 6) + (["dress"] * 2) + (["set"] * 2)
    else:
        route_choices = ["dress", "set", "separates"]
    route_weights = [r for r in route_choices if r != "dress" or dress_pool]
    route_weights = [r for r in route_weights if r != "set" or set_pool]
    route = rng.choice(route_weights or ["separates"])

    if route == "dress":
        add(pick_from(dress_pool), "dress", "bottom")
        if is_cold_context(season, hedef_clo) and outer_pool:
            add(_pick_outer(outer_pool, garments, rng, season, hedef_clo), "outer")
        elif hedef_clo > 0.9 and outer_pool and rng.random() < 0.4:
            add(_pick_outer(outer_pool, garments, rng, season, hedef_clo), "outer")
        add(_pick_footwear(footwear_pool, garments, rng, season, hedef_clo), "footwear")
        if is_cold_context(season, hedef_clo) and acc_pool and rng.random() < 0.45:
            add(pick_from(acc_pool), "accessory")
        return combo

    if route == "set":
        add(pick_from(set_pool), "base", "bottom")
        if is_cold_context(season, hedef_clo) and outer_pool:
            add(_pick_outer(outer_pool, garments, rng, season, hedef_clo), "outer")
        elif hedef_clo > 0.95 and outer_pool and rng.random() < 0.35:
            add(_pick_outer(outer_pool, garments, rng, season, hedef_clo), "outer")
        add(_pick_footwear(footwear_pool, garments, rng, season, hedef_clo), "footwear")
        if is_cold_context(season, hedef_clo) and acc_pool and rng.random() < 0.35:
            add(pick_from(acc_pool), "accessory")
        return combo

    # Ayrı parçalar
    if is_warm_context(season, hedef_clo):
        if base_pool:
            add(pick_from(base_pool), "base")
        elif mid_pool:
            add(pick_from(mid_pool), "mid")
        add(pick_from(bottom_pool), "bottom")
        add(_pick_footwear(footwear_pool, garments, rng, season, hedef_clo), "footwear")
        return combo

    if is_cold_context(season, hedef_clo) and outer_pool:
        add(_pick_outer(outer_pool, garments, rng, season, hedef_clo), "outer")

    if mid_pool:
        add(pick_from(mid_pool), "mid")
    elif base_pool:
        add(pick_from(base_pool), "base")

    add(pick_from(bottom_pool), "bottom")
    add(_pick_footwear(footwear_pool, garments, rng, season, hedef_clo), "footwear")

    if is_cold_context(season, hedef_clo) and acc_pool and rng.random() < 0.45:
        add(pick_from(acc_pool), "accessory")
    elif V_ruzgar >= 15 and acc_pool and rng.random() < 0.3:
        tights = [g for g in acc_pool if garments.get(g, {}).get("subcategory") == "tights"]
        add(pick_from(tights or acc_pool), "accessory")

    return combo


def filter_garments_by_gender(
    garments: dict[str, dict], gender: str | None
) -> dict[str, dict]:
    """Belirli cinsiyet + unisex parçaları döndür (gender None ise hepsi)."""
    if not gender or gender == "all":
        return garments
    return {
        gid: g
        for gid, g in garments.items()
        if g.get("gender", "women") in (gender, "unisex")
    }


def generate_layered_candidates(
    garments: dict[str, dict],
    n_candidates: int = 500,
    season: str | None = None,
    hedef_clo: float = 0.6,
    V_ruzgar: float = 10,
    seed: int = 42,
    gender: str | None = None,
) -> list[list[str]]:
    rng = random.Random(seed)
    garments = filter_garments_by_gender(garments, gender)
    slots = inventory_by_slot(garments, season, hedef_clo)
    pools = _build_combo_pools(slots, garments, season, hedef_clo)
    candidates = []
    attempts = 0
    max_attempts = max(n_candidates * 4, 400)
    while len(candidates) < n_candidates and attempts < max_attempts:
        attempts += 1
        c = build_layered_combo(
            slots, hedef_clo, V_ruzgar, rng, garments, season, pools=pools,
        )
        if len(c) >= 2 and is_valid_outfit_combo(c, garments, season, hedef_clo):
            candidates.append(c)
    return candidates


def thermal_bonuses_penalties(
    pieces: list[dict],
    hedef_clo: float,
    V_ruzgar: float,
    thermal_cats: dict,
    coverage_defaults: dict,
) -> tuple[float, float, float]:
    bonus, penalty = 0.0, 0.0
    total_clo = ensemble_total_clo(pieces, thermal_cats, coverage_defaults)
    subs = {p["subcategory"] for p in pieces}
    slots_present = {garment_slot(p) for p in pieces}
    has_dress = any(is_dress_piece(p) for p in pieces)
    has_set = any(is_coordinated_set(p) for p in pieces)

    if hedef_clo > 1.0:
        if "outer" not in slots_present:
            penalty += 1.5
        if "base" not in slots_present and not has_dress and not has_set:
            penalty += 0.8
        if total_clo < hedef_clo * 0.5:
            penalty += 2.0
        if total_clo >= hedef_clo * 0.7:
            bonus += 0.8
        if subs & {"tank_top", "shorts"}:
            penalty += 2.0
        if "scarf" in subs or "hat" in subs:
            bonus += 0.3

    elif hedef_clo < 0.5:
        if total_clo > 0.55:
            penalty += 1.5
        if subs & {"shorts", "tank_top", "dress_short"}:
            bonus += 0.4
        avg_ret = sum(
            effective_ret(p, thermal_cats, coverage_defaults) for p in pieces
        ) / max(len(pieces), 1)
        if avg_ret > 8:
            penalty += 0.8

    if V_ruzgar >= 15 and "skirt" in subs:
        if "tights" not in subs and "leggings" not in subs:
            penalty += 1.2
        else:
            bonus += 0.4

    return bonus, penalty, total_clo


def style_cohesion_bonus(pieces: list[dict]) -> float:
    """Ana parçalar (üst/alt/dış/elbise) aynı görsel stil kümesindeyse küçük bonus."""
    main = [
        p for p in pieces
        if garment_slot(p) in ("base", "mid", "outer", "bottom", "dress")
        and p.get("style_cluster", -1) >= 0
    ]
    if len(main) < 2:
        return 0.0
    clusters = [p["style_cluster"] for p in main]
    modal = max(set(clusters), key=clusters.count)
    frac = clusters.count(modal) / len(clusters)
    return round(0.5 * frac, 3)


def score_combination(
    piece_ids: list[str],
    garments: dict[str, dict],
    hedef_clo: float,
    V_ruzgar: float,
    thermal_cats: dict,
    coverage_defaults: dict,
    aesthetic_fn,
    season: str | None = None,
) -> dict[str, Any]:
    pieces = [garments[pid] for pid in piece_ids if pid in garments]
    skor_estetik = aesthetic_fn(piece_ids)
    bonus, penalty, total_clo = thermal_bonuses_penalties(
        pieces, hedef_clo, V_ruzgar, thermal_cats, coverage_defaults
    )
    bonus += style_cohesion_bonus(pieces)
    penalty += outfit_coherence_penalty(pieces)
    penalty += season_coherence_penalty(pieces, season, hedef_clo)
    final_skor = skor_estetik + bonus - penalty
    rank = final_skor - abs(hedef_clo - total_clo) * 0.6
    hints = accessory_hints(pieces, hedef_clo, V_ruzgar)
    return {
        "piece_ids": piece_ids,
        "layers": [f"{garment_slot(p)}:{p['name']}" for p in pieces],
        "skor_estetik": round(skor_estetik, 3),
        "bonus": round(bonus, 3),
        "penalty": round(penalty, 3),
        "final_skor": round(final_skor, 3),
        "total_Clo_C": total_clo,
        "delta_Clo": round(abs(hedef_clo - total_clo), 4),
        "rank": round(rank, 3),
        "accessory_hints": hints,
    }
