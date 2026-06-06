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


def inventory_by_slot(
    garments: dict[str, dict], season: str | None = None
) -> dict[str, list[str]]:
    """Envanteri katman slotlarına göre grupla."""
    slots: dict[str, list[str]] = {s: [] for s in LAYER_SLOTS}
    for gid, g in garments.items():
        if season and season not in g.get("season_usable", [season]):
            continue
        slot = garment_slot(g)
        if slot in slots:
            slots[slot].append(gid)
    return slots


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


def build_layered_combo(
    slots: dict[str, list[str]],
    hedef_clo: float,
    V_ruzgar: float,
    rng: random.Random,
) -> list[str]:
    """
    Katmanlı kombin üret:
    base (tişört/termal) + mid (kazak) + outer (mont) + bottom + aksesuarlar
    """
    combo: list[str] = []
    used_slots: set[str] = set()

    def pick(slot: str, prob: float) -> None:
        if slot in used_slots or not slots.get(slot):
            return
        if rng.random() < prob:
            combo.append(rng.choice(slots[slot]))
            used_slots.add(slot)

    # Elbise rotası
    if slots.get("dress") and rng.random() < 0.12:
        combo.append(rng.choice(slots["dress"]))
        used_slots.update({"dress", "bottom"})
        pick("outer", 0.55 if hedef_clo > 0.9 else 0.15)
        pick("footwear", 0.85)
        pick("accessory", 0.7 if hedef_clo > 1.0 else 0.2)
        return combo

    # Soğuk: iç katman zorunluluğa yakın
    if hedef_clo > 1.2:
        pick("base", 0.85)
    elif hedef_clo > 0.7:
        pick("base", 0.55)
    else:
        pick("base", 0.25)

    pick("mid", 0.90 if hedef_clo > 0.8 else 0.45)
    pick("outer", min(0.95, 0.35 + hedef_clo * 0.45))
    pick("bottom", 0.98)
    pick("footwear", 0.80)

    # Aksesuarlar
    if hedef_clo > 1.0:
        pick("accessory", 0.75)  # atkı vb.
        if rng.random() < 0.5:
            # ikinci aksesuar şansı (şapka) — aynı slottan farklı parça
            hats = [i for i in slots.get("accessory", []) if True]
            if hats and len(combo) < 7:
                extra = rng.choice(hats)
                if extra not in combo:
                    combo.append(extra)

    # Rüzgarlı + etek → tayt/kilotlu çorap slotu (leggings veya tights)
    if combo:
        piece_map = {gid: gid for gid in combo}
        # bottom parçayı bul
        bottom_ids = [c for c in combo]  # caller'da garments ile kontrol edilecek
        if V_ruzgar >= 15 and hedef_clo < 1.1:
            pick("accessory", 0.65)

    return combo


def generate_layered_candidates(
    garments: dict[str, dict],
    n_candidates: int = 500,
    season: str | None = None,
    hedef_clo: float = 0.6,
    V_ruzgar: float = 10,
    seed: int = 42,
) -> list[list[str]]:
    rng = random.Random(seed)
    slots = inventory_by_slot(garments, season)
    candidates = []
    for _ in range(n_candidates):
        c = build_layered_combo(slots, hedef_clo, V_ruzgar, rng)
        if len(c) >= 2:
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

    if hedef_clo > 1.0:
        if "outer" not in slots_present:
            penalty += 1.5
        if "base" not in slots_present:
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


def score_combination(
    piece_ids: list[str],
    garments: dict[str, dict],
    hedef_clo: float,
    V_ruzgar: float,
    thermal_cats: dict,
    coverage_defaults: dict,
    aesthetic_fn,
) -> dict[str, Any]:
    pieces = [garments[pid] for pid in piece_ids if pid in garments]
    skor_estetik = aesthetic_fn(piece_ids)
    bonus, penalty, total_clo = thermal_bonuses_penalties(
        pieces, hedef_clo, V_ruzgar, thermal_cats, coverage_defaults
    )
    final_skor = skor_estetik + bonus - penalty
    rank = final_skor - abs(hedef_clo - total_clo)
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
