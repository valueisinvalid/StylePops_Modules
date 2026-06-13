"""StylePops görsel envanter — Streamlit arayüzü."""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Mac: PyTorch + LightGBM/OpenMP çakışmasını azalt
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aesthetic_compatibility import aesthetic_compatibility_score, load_garments, make_aesthetic_fn
from stylepops_core import (
    apparent_temperature,
    generate_layered_candidates,
    interpolate_hedef_clo,
    is_valid_outfit_combo,
    score_combination,
)

VISUAL = ROOT / "data" / "visual"
LOOKUPS = ROOT / "data" / "lookups"
PREFS_LOG = VISUAL / "preferences_log.csv"
COMBO_SEL_LOG = VISUAL / "combo_selections.csv"


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


def _image_width() -> str:
    """Streamlit 1.50+ width API; eski sürümlerde use_container_width."""
    try:
        from streamlit import __version__ as st_ver
        major, minor = (int(x) for x in st_ver.split(".")[:2])
        if (major, minor) >= (1, 50):
            return "stretch"
    except Exception:
        pass
    return "stretch"


def st_image(path: str, width: int | None = None) -> None:
    if width is not None:
        try:
            st.image(path, width=width)
            return
        except TypeError:
            st.image(path)
            return
    w = _image_width()
    try:
        st.image(path, width=w)
    except TypeError:
        st.image(path, use_container_width=True)


def season_map(scenario_id: str) -> str:
    return {
        "kis_soguk_ruzgarli": "kis",
        "sonbahar_serin": "sonbahar",
        "ilkbahar_ilik": "ilkbahar",
        "yaz_sicak": "yaz",
    }.get(scenario_id, "ilkbahar")


GENDER_LABELS = {"women": "Kadın", "men": "Erkek", "unisex": "Unisex"}


def _gender_counts(garments: dict) -> dict:
    counts = {"women": 0, "men": 0, "unisex": 0}
    for g in garments.values():
        counts[g.get("gender", "women")] = counts.get(g.get("gender", "women"), 0) + 1
    return counts


def render_inventory(garments: dict) -> None:
    counts = _gender_counts(garments)
    st.subheader(f"Gardırop ({len(garments)} parça)")
    st.caption(
        f"Kadın: {counts.get('women', 0)} · Erkek: {counts.get('men', 0)} · "
        f"Unisex: {counts.get('unisex', 0)}"
    )

    gender_opts = ["Kadın", "Erkek", "Unisex"]
    gender_sel = st.radio("Cinsiyet", gender_opts, horizontal=True, index=0)
    gender_key = {"Kadın": "women", "Erkek": "men", "Unisex": "unisex"}[gender_sel]

    in_gender = [
        g for g in garments.values()
        if g.get("gender", "women") == gender_key
        or (gender_key in ("women", "men") and g.get("gender") == "unisex")
    ]
    categories = sorted({g["category"] for g in in_gender})
    cat_filter = st.multiselect("Kategori", categories, default=categories)
    season_filter = st.selectbox("Mevsim", ["tümü", "kis", "sonbahar", "ilkbahar", "yaz"])

    cluster_labels = sorted({
        g.get("style_cluster_label") for g in in_gender if g.get("style_cluster_label")
    })
    style_filter = "tümü"
    if cluster_labels:
        style_filter = st.selectbox("Stil kümesi", ["tümü", *cluster_labels])

    shown = 0
    cols = st.columns(4)
    for g in sorted(in_gender, key=lambda x: x["id"]):
        if g["category"] not in cat_filter:
            continue
        if season_filter != "tümü" and season_filter not in g.get("season_usable", []):
            continue
        if style_filter != "tümü" and g.get("style_cluster_label") != style_filter:
            continue
        img_path = ROOT / g.get("image_path", "")
        with cols[shown % 4]:
            if img_path.exists():
                st_image(str(img_path))
            else:
                st.caption("(görsel yok)")
            st.caption(f"**{g['name'][:40]}**")
            tag = GENDER_LABELS.get(g.get("gender", "women"), "")
            st.caption(f"{g['id']} · {g['subcategory']} · {tag}")
        shown += 1
    st.caption(f"Gösterilen: {shown} parça")


def render_recommendations(garments: dict) -> None:
    if not garments:
        st.warning("Önce `python scripts/run_visual_pipeline.py --all` çalıştırın.")
        return

    thermal_cats, coverage, clo_points, scenarios = load_lookups()
    scenario_id = st.selectbox("Senaryo", list(scenarios.keys()),
                               format_func=lambda x: scenarios[x]["label_tr"])
    top_k = st.slider("Top öneri", 1, 10, 3)

    scenario = scenarios[scenario_id]
    season = season_map(scenario_id)
    T_app = apparent_temperature(scenario["T_hava"], scenario["RH_nem"], scenario["V_ruzgar"])
    hedef = interpolate_hedef_clo(T_app, clo_points)

    st.info(f"T_hissedilen={T_app}°C · hedef_Clo={hedef}")

    if not st.button("Önerileri hesapla", type="primary"):
        st.caption("FashionCLIP yüklemesi ve skorlama yalnızca butona basınca çalışır.")
        return

    aesthetic_fn = make_aesthetic_fn(garments)
    with st.spinner("Kombinler skorlanıyor (FashionCLIP + termal)…"):
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
                thermal_cats, coverage, aesthetic_fn, season=season,
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
                    st_image(str(p))
        st.caption(
            f"estetik={r['aesthetic_score']} · FashionCLIP={r.get('fashionclip_score')} · "
            f"Clo={r['total_Clo_C']} · ΔClo={r['delta_Clo']}"
        )
        st.divider()


AB_BATCH_SIZE = 50


def _rated_pair_ids() -> set[str]:
    if not PREFS_LOG.exists():
        return set()
    with PREFS_LOG.open(encoding="utf-8") as f:
        return {row["pair_id"] for row in csv.DictReader(f) if row.get("pair_id")}


def _init_ab_session(pairs: list[dict]) -> None:
    if "ab_initialized" not in st.session_state:
        st.session_state.ab_initialized = True
        st.session_state.ab_batch = 0
        st.session_state.ab_index = 0
        st.session_state.ab_batch_done = False
        st.session_state.ab_queue = []


def _combo_row_valid(combo: dict, garments: dict) -> bool:
    piece_ids = [p for p in combo.get("piece_ids", "").split("|") if p]
    if len(piece_ids) < 2:
        return False
    return is_valid_outfit_combo(
        piece_ids,
        garments,
        combo.get("season"),
        float(combo.get("hedef_Clo", 0.9)),
    )


def _filter_valid_ab_pairs(pairs: list[dict], combos: list[dict], garments: dict) -> list[dict]:
    by_id = {c["combo_id"]: c for c in combos}
    valid = []
    for pair in pairs:
        a = by_id.get(pair["combo_a_id"])
        b = by_id.get(pair["combo_b_id"])
        if a and b and _combo_row_valid(a, garments) and _combo_row_valid(b, garments):
            valid.append(pair)
    return valid


def _refresh_ab_queue(pairs: list[dict], combos: list[dict], garments: dict) -> None:
    rated = _rated_pair_ids()
    remaining = [p for p in pairs if p["pair_id"] not in rated]
    remaining = _filter_valid_ab_pairs(remaining, combos, garments)
    import random
    rng = random.Random(42 + st.session_state.ab_batch)
    pool = remaining[:]
    rng.shuffle(pool)
    start = st.session_state.ab_batch * AB_BATCH_SIZE
    st.session_state.ab_queue = pool[start:start + AB_BATCH_SIZE]
    st.session_state.ab_index = 0
    st.session_state.ab_batch_done = False


def render_ab_test(pairs: list[dict], garments: dict) -> None:
    combos = load_combos()
    pairs = _filter_valid_ab_pairs(pairs, combos, garments)

    gkey = "women"
    if any(p.get("gender") for p in pairs):
        gsel = st.radio("Cinsiyet", ["Kadın", "Erkek"], horizontal=True, key="ab_gender")
        gkey = {"Kadın": "women", "Erkek": "men"}[gsel]
        pairs = [p for p in pairs if p.get("gender", "women") == gkey]

    if not pairs:
        st.warning("Bu cinsiyet için geçerli A/B çifti yok.")
        return

    _init_ab_session(pairs)

    if st.session_state.get("ab_active_gender") != gkey:
        st.session_state.ab_active_gender = gkey
        st.session_state.ab_batch = 0
        _refresh_ab_queue(pairs, combos, garments)

    rater_id = st.text_input("Değerlendirici ID (isteğe bağlı)", key="ab_rater_id")

    if st.button("A/B turunu sıfırla"):
        for key in list(st.session_state.keys()):
            if key.startswith("ab_"):
                del st.session_state[key]
        st.rerun()

    if st.session_state.ab_batch_done:
        st.success(f"Bu turda {len(st.session_state.ab_queue)} karşılaştırma tamamlandı.")
        rated_total = len(_rated_pair_ids())
        st.caption(f"Toplam kayıtlı tercih: {rated_total}")
        if st.button("50 karşılaştırma daha", type="primary"):
            st.session_state.ab_batch += 1
            _refresh_ab_queue(pairs, combos, garments)
            st.rerun()
        return

    if not st.session_state.ab_queue:
        _refresh_ab_queue(pairs, combos, garments)

    queue = st.session_state.ab_queue
    if not queue:
        st.info("Tüm A/B çiftleri değerlendirildi.")
        return

    idx = st.session_state.ab_index
    if idx >= len(queue):
        st.session_state.ab_batch_done = True
        st.rerun()

    pair = queue[idx]
    st.progress((idx + 1) / len(queue), text=f"Tur {st.session_state.ab_batch + 1} · {idx + 1}/{len(queue)}")
    st.caption(f"{pair['pair_id']} · {pair['scenario_id']}")

    collage = ROOT / pair["collage_path"]
    if collage.exists():
        st_image(str(collage), width=760)
    else:
        col1, col2 = st.columns(2)
        combos_by_id = {c["combo_id"]: c for c in load_combos()}
        for col, cid, label in ((col1, pair["combo_a_id"], "A"), (col2, pair["combo_b_id"], "B")):
            with col:
                st.markdown(f"**{label}** — {cid}")
                combo = combos_by_id.get(cid)
                if combo:
                    for pid in [p for p in combo["piece_ids"].split("|") if p]:
                        g = garments.get(pid)
                        if g and (ROOT / g.get("image_path", "")).exists():
                            st_image(str(ROOT / g["image_path"]))

    def _vote(winner: str) -> None:
        save_preference(pair["pair_id"], winner, rater_id)
        st.session_state.ab_index += 1
        st.rerun()

    c1, c2, c3 = st.columns(3)
    if c1.button("A daha iyi", use_container_width=True, key=f"ab_a_{pair['pair_id']}"):
        _vote("A")
    if c2.button("B daha iyi", use_container_width=True, key=f"ab_b_{pair['pair_id']}"):
        _vote("B")
    if c3.button("Berabere", use_container_width=True, key=f"ab_tie_{pair['pair_id']}"):
        _vote("tie")

    rated_total = len(_rated_pair_ids())
    st.caption(f"Toplam kayıtlı tercih: {rated_total}")


def save_combo_selection(combo_id: str, action: str, scenario: str, gender: str, rater: str) -> None:
    VISUAL.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "combo_id": combo_id,
        "action": action,
        "scenario_id": scenario,
        "gender": gender,
        "rater_id": rater or "anonymous",
    }
    write_header = not COMBO_SEL_LOG.exists()
    with COMBO_SEL_LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _selected_combo_ids() -> set[str]:
    if not COMBO_SEL_LOG.exists():
        return set()
    with COMBO_SEL_LOG.open(encoding="utf-8") as f:
        return {r["combo_id"] for r in csv.DictReader(f) if r.get("action") == "like"}


COMBO_SELECT_PAGE = 12


def render_combo_select(combos: list[dict], garments: dict) -> None:
    """Kurallar + FashionCLIP'ten geçmiş kombinleri kullanıcıya seçtir."""
    if not combos:
        st.warning("Önce kombin üretin: `python scripts/generate_visual_combinations.py`")
        return

    st.subheader("Kombin Seç")
    st.caption("Nesnel kurallar + FashionCLIP'ten geçen kombinler. Beğendiklerini işaretle.")

    gsel = st.radio("Cinsiyet", ["Kadın", "Erkek"], horizontal=True, key="sel_gender")
    gkey = {"Kadın": "women", "Erkek": "men"}[gsel]
    scenarios = sorted({c["scenario_id"] for c in combos})
    scenario = st.selectbox("Senaryo", scenarios, key="sel_scenario")
    rater = st.text_input("Değerlendirici ID (isteğe bağlı)", key="sel_rater")

    pool = [
        c for c in combos
        if c["scenario_id"] == scenario
        and c.get("gender", "women") == gkey
        and _combo_row_valid(c, garments)
    ]
    pool.sort(key=lambda x: float(x["rank"]), reverse=True)

    if not pool:
        st.info("Bu cinsiyet/senaryo için geçerli kombin yok.")
        return

    page_key = f"sel_page_{gkey}_{scenario}"
    page = st.session_state.get(page_key, 0)
    start = page * COMBO_SELECT_PAGE
    batch = pool[start:start + COMBO_SELECT_PAGE]
    liked = _selected_combo_ids()

    st.caption(f"{len(pool)} kombin · sayfa {page + 1}/{(len(pool) - 1) // COMBO_SELECT_PAGE + 1} · beğenilen: {len(liked & {c['combo_id'] for c in pool})}")

    cols = st.columns(3)
    for i, c in enumerate(batch):
        with cols[i % 3]:
            collage = ROOT / c["collage_path"]
            if collage.exists():
                st_image(str(collage))
            else:
                names = ", ".join(
                    garments[p]["name"][:20]
                    for p in c["piece_ids"].split("|") if p in garments
                )
                st.caption(names)
            cid = c["combo_id"]
            already = cid in liked
            st.caption(f"{cid} · Rank={c['rank']} · FC={c.get('fashionclip_score')}")
            bc1, bc2 = st.columns(2)
            if bc1.button("❤ Beğen" if not already else "✓ Beğenildi",
                          key=f"like_{cid}", use_container_width=True, disabled=already):
                save_combo_selection(cid, "like", scenario, gkey, rater)
                st.rerun()
            if bc2.button("Geç", key=f"skip_{cid}", use_container_width=True):
                save_combo_selection(cid, "skip", scenario, gkey, rater)
                st.rerun()

    nav1, nav2 = st.columns(2)
    if page > 0 and nav1.button("← Önceki", use_container_width=True):
        st.session_state[page_key] = page - 1
        st.rerun()
    if start + COMBO_SELECT_PAGE < len(pool) and nav2.button("Sonraki →", use_container_width=True):
        st.session_state[page_key] = page + 1
        st.rerun()


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
                st_image(str(collage))
        with cols[1]:
            st.markdown(f"**{c['combo_id']}** · Rank={c['rank']}")
            st.caption(
                f"estetik={c['aesthetic_score']} · FC={c.get('fashionclip_score')} · "
                f"ΔClo={c['delta_Clo']} · {c['layer_count']} parça"
            )


def render_status() -> None:
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

    page = st.sidebar.radio(
        "Sayfa",
        ["Gardırop", "Canlı Öneri", "Kombin Seç", "Kayıtlı Kombinler", "A/B Tercih", "Durum"],
        index=0,
    )

    if page == "Gardırop":
        render_inventory(garments)
    elif page == "Canlı Öneri":
        render_recommendations(garments)
    elif page == "Kombin Seç":
        render_combo_select(combos, garments)
    elif page == "Kayıtlı Kombinler":
        render_saved_combos(combos, garments)
    elif page == "A/B Tercih":
        render_ab_test(pairs, garments)
    else:
        render_status()


if __name__ == "__main__":
    main()
