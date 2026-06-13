#!/usr/bin/env python3
"""FashionCLIP embedding'lerinden görsel stil kümeleri üretir (KMeans).

- data/visual/clip_embeddings.npz okunur
- Her parçaya `style_cluster` (int) ve `style_cluster_label` (insan-okur) atanır
- garments_production.json güncellenir + data/visual/style_clusters.json yazılır

Kümeler, benzer görünen parçaları (ör. atletik üstler, kışlık montlar, zarif
elbiseler) gruplar; kombin üretiminde stil-uyumu bonusu ve Streamlit filtresi
için kullanılır.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PROD = ROOT / "data" / "visual" / "garments_production.json"
CACHE = ROOT / "data" / "visual" / "clip_embeddings.npz"
OUT_CLUSTERS = ROOT / "data" / "visual" / "style_clusters.json"

DEFAULT_K = 16


def _label_for_cluster(members: list[dict]) -> str:
    from stylepops_core import garment_slot, garment_style
    slots = Counter(garment_slot(m) for m in members)
    styles = Counter(garment_style(m) for m in members)
    subs = Counter(m.get("subcategory", "") for m in members)
    top_slot = slots.most_common(1)[0][0] if slots else "?"
    top_style = styles.most_common(1)[0][0] if styles else "casual"
    top_sub = subs.most_common(1)[0][0] if subs else "?"
    style_tr = {"sport": "spor", "formal": "formal", "casual": "günlük"}.get(top_style, top_style)
    return f"{style_tr}-{top_slot}-{top_sub}"


def main(k: int = DEFAULT_K) -> int:
    from sklearn.cluster import KMeans

    if not CACHE.exists():
        print("HATA: clip_embeddings.npz yok — önce precompute_clip_cache.py", file=sys.stderr)
        return 1

    data = json.loads(PROD.read_text(encoding="utf-8"))
    garments = data["garments"]
    cache = np.load(CACHE)
    cached_ids = set(cache.files)

    ids = [g["id"] for g in garments if g["id"] in cached_ids]
    if not ids:
        print("HATA: embedding eşleşmesi yok", file=sys.stderr)
        return 1
    X = np.stack([cache[i] for i in ids])

    k = min(k, len(ids))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    id_to_cluster = dict(zip(ids, labels.tolist()))

    by_cluster: dict[int, list[dict]] = {}
    for g in garments:
        c = id_to_cluster.get(g["id"])
        g["style_cluster"] = int(c) if c is not None else -1
        if c is not None:
            by_cluster.setdefault(int(c), []).append(g)

    cluster_labels = {}
    for c, members in sorted(by_cluster.items()):
        label = _label_for_cluster(members)
        cluster_labels[str(c)] = {"label": label, "count": len(members)}
        for g in members:
            g["style_cluster_label"] = label

    PROD.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_CLUSTERS.write_text(
        json.dumps({"k": k, "clusters": cluster_labels}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"{k} stil kümesi atandı → {len(ids)} parça")
    for c in sorted(cluster_labels, key=lambda x: int(x)):
        info = cluster_labels[c]
        print(f"  küme {c:>2}: {info['count']:>4} parça · {info['label']}")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=DEFAULT_K)
    args = ap.parse_args()
    raise SystemExit(main(args.k))
