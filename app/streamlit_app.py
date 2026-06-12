"""StylePops görsel envanter — Streamlit arayüzü."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aesthetic_compatibility import aesthetic_compatibility_score, load_garments, make_aesthetic_fn
from stylepops_core import (
    apparent_temperature,
    generate_layered_candidates,
    interpolate_hedef_clo,
    score_combination,
)

VISUAL = ROOT / "data" / "visual"
LOOKUPS = ROOT / "data" / "lookups"
PREFS_LOG = VISUAL / "preferences_log.csv"


@st.cache_data
def load_garments_cached() -> dict:
    sys.path.insert(0, str(ROOT / "scripts"))
    from inventory_loader import load_production_garments, catalog_meta

    meta = catalog_meta("production")
    if not meta.get("exists"):
        return {}
    return load_production_garments()


@st.cache_data
def load_combos() -> list[dict]:
    path = VISUAL / "combinations_visual.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@st.cache_data
def load_ab_pairs() -> list[dict]:
    path = VISUAL / "ab_pairs.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_lookups():
    with (LOOKUPS / "fabric_properties.json").open(encoding="utf-8") as f:
        fab = json.load(f)
    with (LOOKUPS / "coverage_ratios.json").open(encoding="utf-8") as f:
        cov = json.load(f)
    with (LOOKUPS / "target_clo_points.json").open(encoding="utf-8") as f:
        tgt = json.load(f)
    return (
        fab["thermal_categories"],
        cov["coverage_by_subcategory"],
        tgt["target_clo_points"],
        tgt["weather_scenarios"],
    )


def save_preference(pair_id: str, winner: str, rater_id: str) -> None:
    VISUAL.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pair_id": pair_id,
        "preference_winner": winner,
        "rater_id": rater_id or "anonymous",
    }
    write_header = not PREFS_LOG.exists()
    with PREFS_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def season_map(scenario_id: str) -> str:
    return {
        "kis_soguk_ruzgarli": "kis",
        "sonbahar_serin": "sonbahar",
        "ilkbahar_ilik": "ilkbahar",
        "yaz_sicak": "yaz",
    }.get(scenario_id, "ilkbahar")


def render_inventory(garments: dict) -> None:
    st.subheader(f"Gardırop ({len(garments)} parça)")
    categories = sorted({g["category"] for g in garments.values()})
    cat_filter = st.multiselect("Kategori", categories, default=categories)
    season_filter = st.selectbox("Mevsim", ["tümü", "kis", "sonbahar", "ilkbahar", "yaz"])

    cols = st.columns(4)
    i = 0
    for g in sorted(garments.values(), key=lambda x: x["id"]):
        if g["category"] not in cat_filter:
            continue
        if season_filter != "tümü" and season_filter not in g.get("season_usable", []):
            continue
        img_path = ROOT / g.get("image_path", "")
        with cols[i % 4]:
            if img_path.exists():
                st.image(str(img_path), use_container_width=True)
            else:
                st.caption("(görsel yok)")
            st.caption(f"**{g['name'][:40]}**")
            st.caption(f"{g['id']} · {g['subcategory']}")
        i += 1


def render_recommendations(garments: dict) -> None:
    if not garments:
        st.warning("Önce `python scripts/run_visual_pipeline.py --all` çalıştırın.")
        return

    thermal_cats, coverage, clo_points, scenarios = load_lookups()
    aesthetic_fn = make_aesthetic_fn(garments)
    scenario_id = st.selectbox("Senaryo", list(scenarios.keys()),
                               format_func=lambda x: scenarios[x]["label_tr"])
    top_k = st.slider("Top öneri", 1, 10, 3)

    scenario = scenarios[scenario_id]
    season = season_map(scenario_id)
    T_app = apparent_temperature(scenario["T_hava"], scenario["RH_nem"], scenario["V_ruzgar"])
    hedef = interpolate_hedef_clo(T_app, clo_points)

    st.info(f"T_hissedilen={T_app}°C · hedef_Clo={hedef}")

    candidates = generate_layered_candidates(
        garments, n_candidates=600, season=season,
        hedef_clo=hedef, V_ruzgar=scenario["V_ruzgar"], seed=42,
    )
    seen = set()
    results = []
    for piece_ids in candidates:
        key = tuple(sorted(piece_ids))
        if key in seen:
            continue
        seen.add(key)
        r = score_combination(
            piece_ids, garments, hedef, scenario["V_ruzgar"],
            thermal_cats, coverage, aesthetic_fn,
        )
        aes = aesthetic_compatibility_score(piece_ids, garments)
        r.update(aes)
        results.append(r)
    results.sort(key=lambda x: x["rank"], reverse=True)

    for i, r in enumerate(results[:top_k], 1):
        names = " + ".join(garments[pid]["name"] for pid in r["piece_ids"] if pid in garments)
        st.markdown(f"**#{i} Rank={r['rank']}** · {len(r['piece_ids'])} parça")
        st.caption(names)
        cols = st.columns(min(6, len(r["piece_ids"])))
        for j, pid in enumerate(r["piece_ids"]):
            g = garments.get(pid)
            if not g:
                continue
            p = ROOT / g.get("image_path", "")
            with cols[j % len(cols)]:
                if p.exists():
                    st.image(str(p), use_container_width=True)
        st.caption(
            f"estetik={r['aesthetic_score']} · FashionCLIP={r.get('fashionclip_score')} · "
            f"Clo={r['total_Clo_C']} · ΔClo={r['delta_Clo']}"
        )
        st.divider()


def render_ab_test(pairs: list[dict], garments: dict) -> None:
    if not pairs:
        st.warning("A/B çiftleri yok. Pipeline `--combinations` adımını çalıştırın.")
        return

    rater_id = st.text_input("Değerlendirici ID (isteğe bağlı)", "")
    pair = st.selectbox("Karşılaştırma", pairs, format_func=lambda p: f"{p['pair_id']} · {p['scenario_id']}")

    collage = ROOT / pair["collage_path"]
    if collage.exists():
        st.image(str(collage), use_container_width=True)
    else:
        col1, col2 = st.columns(2)
        for col, cid, label in ((col1, pair["combo_a_id"], "A"), (col2, pair["combo_b_id"], "B")):
            with col:
                st.markdown(f"**{label}** — {cid}")

    c1, c2, c3 = st.columns(3)
    if c1.button("A daha iyi", use_container_width=True):
        save_preference(pair["pair_id"], "A", rater_id)
        st.success("Kaydedildi: A")
    if c2.button("B daha iyi", use_container_width=True):
        save_preference(pair["pair_id"], "B", rater_id)
        st.success("Kaydedildi: B")
    if c3.button("Berabere", use_container_width=True):
        save_preference(pair["pair_id"], "tie", rater_id)
        st.success("Kaydedildi: berabere")

    if PREFS_LOG.exists():
        with PREFS_LOG.open(encoding="utf-8") as f:
            n = sum(1 for _ in csv.DictReader(f))
        st.caption(f"Toplam kayıtlı tercih: {n}")


def render_saved_combos(combos: list[dict], garments: dict) -> None:
    if not combos:
        return
    scenario = st.selectbox("Senaryo filtresi", sorted({c["scenario_id"] for c in combos}))
    filtered = [c for c in combos if c["scenario_id"] == scenario]
    filtered.sort(key=lambda x: float(x["rank"]), reverse=True)

    for c in filtered[:8]:
        collage = ROOT / c["collage_path"]
        cols = st.columns([1, 2])
        with cols[0]:
            if collage.exists():
                st.image(str(collage), use_container_width=True)
        with cols[1]:
            st.markdown(f"**{c['combo_id']}** · Rank={c['rank']}")
            st.caption(
                f"estetik={c['aesthetic_score']} · FC={c.get('fashionclip_score')} · "
                f"ΔClo={c['delta_Clo']} · {c['layer_count']} parça"
            )


def main() -> None:
    st.set_page_config(page_title="StylePops Visual", layout="wide")
    st.title("StylePops — Görsel Envanter")
    st.caption("Livostyle MIT · FashionCLIP estetik · Katmanlı termal öneri")

    garments = load_garments_cached()
    combos = load_combos()
    pairs = load_ab_pairs()

    if not garments:
        st.error(
            "Görsel envanter bulunamadı. Terminalde:\n\n"
            "`python scripts/run_visual_pipeline.py --all`"
        )
        st.stop()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Gardırop", "Canlı Öneri", "Kayıtlı Kombinler", "A/B Tercih", "Durum",
    ])

    with tab1:
        render_inventory(garments)
    with tab2:
        render_recommendations(garments)
    with tab3:
        render_saved_combos(combos, garments)
    with tab4:
        render_ab_test(pairs, garments)
    with tab5:
        manifest_path = VISUAL / "manifest.json"
        if manifest_path.exists():
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")))
        st.markdown("Detaylı takip: `PROJE_DURUMU.md` · Lisans: `DATA_PROVENANCE.md`")
        status_path = VISUAL / "status.json"
        if status_path.exists():
            st.subheader("Pipeline durumu")
            st.json(json.loads(status_path.read_text(encoding="utf-8")))
        sys.path.insert(0, str(ROOT / "scripts"))
        from inventory_loader import catalog_meta
        st.subheader("Envanter kayıtları")
        st.json({"production": catalog_meta("production"), "training_44k": catalog_meta("training")})


if __name__ == "__main__":
    main()
