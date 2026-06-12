#!/usr/bin/env python3
"""
Outfit uyumluluk başlığı (compatibility head) — 44K MIT korpusundan sentetik çiftler.
Frozen/Lora FashionCLIP embedding + LightGBM regresyon → kombin estetik skoru.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aesthetic_compatibility import garment_image_embedding
from inventory_loader import load_training_garments

OUT_PATH = ROOT / "outputs" / "compatibility_head_v1.joblib"

SLOT_COMPATIBLE = {
    "top": {"bottom", "dress"},
    "outer": {"top", "bottom", "dress"},
    "bottom": {"top", "outer"},
    "dress": {"outer", "footwear"},
    "footwear": {"top", "bottom", "dress"},
}


def outfit_compatible(pieces: list[dict]) -> bool:
    roles = {p["layer_role"] for p in pieces}
    cats = {p["category"] for p in pieces}
    if "dress" in roles:
        return len(pieces) >= 1
    if "top" in cats or any(p["layer_role"] in ("base", "mid") for p in pieces):
        return "bottom" in cats or "dress" in roles
    return len(pieces) >= 2


def build_pairs(garments: dict, n_pos: int, n_neg: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    by_bucket: dict[str, list[dict]] = {}
    for g in garments.values():
        key = f"{g['fp_meta']['gender']}|{g['fp_meta']['usage']}|{g['season_primary']}|{g['layer_role']}"
        by_bucket.setdefault(key, []).append(g)

    X, y = [], []

    def embed_combo(items: list[dict]) -> np.ndarray | None:
        embs = []
        for g in items:
            e = garment_image_embedding(g)
            if e is not None:
                embs.append(e)
        if len(embs) < 2:
            return None
        mat = np.stack(embs)
        # outfit feature: mean + std + max pairwise sim
        mean = mat.mean(axis=0)
        std = mat.std(axis=0)
        sims = [float(np.dot(mat[i], mat[j])) for i in range(len(mat)) for j in range(i + 1, len(mat))]
        feat = np.concatenate([mean, std, [np.mean(sims), np.min(sims), np.max(sims)]])
        return feat

    # positive pairs: same bucket, different layer roles
    attempts = 0
    while len(y) < n_pos and attempts < n_pos * 50:
        attempts += 1
        bucket = rng.choice(list(by_bucket.keys()))
        pool = by_bucket[bucket]
        if len(pool) < 2:
            continue
        a, b = rng.sample(pool, 2)
        if a["layer_role"] == b["layer_role"]:
            continue
        pieces = [a, b]
        if not outfit_compatible(pieces):
            continue
        feat = embed_combo(pieces)
        if feat is None:
            continue
        X.append(feat)
        y.append(1.0)

    attempts = 0
    while sum(1 for v in y if v < 0.5) < n_neg and attempts < n_neg * 50:
        attempts += 1
        g1 = rng.choice(list(garments.values()))
        g2 = rng.choice(list(garments.values()))
        if g1["id"] == g2["id"]:
            continue
        if g1["fp_meta"]["gender"] != g2["fp_meta"]["gender"]:
            label = 0.0
        elif g1["layer_role"] == g2["layer_role"] and g1["category"] == g2["category"]:
            label = 0.0
        else:
            label = 0.0 if rng.random() < 0.7 else 1.0
        feat = embed_combo([g1, g2])
        if feat is None:
            continue
        X.append(feat)
        y.append(label)

    return np.array(X), np.array(y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos", type=int, default=2000)
    parser.add_argument("--neg", type=int, default=2000)
    parser.add_argument("--embedding-sample", type=int, default=5000, help="Embedding önbellek için max parça")
    args = parser.parse_args()

    try:
        import joblib
        import lightgbm as lgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("pip install lightgbm scikit-learn joblib", file=sys.stderr)
        sys.exit(1)

    garments = load_training_garments()
    if not garments:
        print("Önce 44K import: python scripts/import_fashion_product_images.py", file=sys.stderr)
        sys.exit(1)

    # embedding warmup on subset
    sample_ids = list(garments.keys())[: args.embedding_sample]
    ok = 0
    for gid in sample_ids:
        if garment_image_embedding(garments[gid]) is not None:
            ok += 1
    print(f"Embedding hazır: {ok}/{len(sample_ids)} (örneklem)")

    X, y = build_pairs(garments, args.pos, args.neg, seed=42)
    print(f"Eğitim çiftleri: {len(y)} (pozitif={int(y.sum())}, negatif={int((1-y).sum())})")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42)
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, prob)
    print(f"Uyumluluk AUC: {auc:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "auc": auc, "feature_dim": X.shape[1]}, OUT_PATH)
    print(f"Kaydedildi → {OUT_PATH}")


if __name__ == "__main__":
    main()
