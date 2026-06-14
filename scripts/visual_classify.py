#!/usr/bin/env python3
"""Önbellekteki FashionCLIP görsel embedding'lerinden GÖRSEL kategori çıkarımı.

torch GEREKTİRMEZ — `build_clip_text_labels.py` ile önceden hesaplanmış metin
etiketi embedding'lerini ve `clip_embeddings.npz` görsel embedding'lerini
kullanarak kosinüs benzerliğiyle her parçayı sınıflandırır.

Kullanım:
    vc = VisualClassifier.load()
    pred = vc.predict_id("FN0884")   # -> {macro, layer_role, subcategory, score, margin}
    pred = vc.predict_vec(vec)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEXT_LABELS = ROOT / "data" / "visual" / "clip_text_labels.npz"
IMG_CACHE = ROOT / "data" / "visual" / "clip_embeddings.npz"


class VisualClassifier:
    def __init__(self, labels: dict, img_cache: dict[str, np.ndarray]):
        self.prompts = list(labels["prompts"])
        self.layer_role = list(labels["layer_role"])
        self.category = list(labels["category"])
        self.subcategory = list(labels["subcategory"])
        self.macro = list(labels["macro"])
        self.T = labels["embeddings"].astype(np.float32)  # [L,512] L2-normalize
        # normalize garanti
        self.T /= (np.linalg.norm(self.T, axis=1, keepdims=True) + 1e-8)
        self.img = img_cache

    @classmethod
    def load(cls) -> "VisualClassifier":
        lbl = np.load(TEXT_LABELS, allow_pickle=True)
        img = {}
        if IMG_CACHE.exists():
            d = np.load(IMG_CACHE, allow_pickle=True)
            for k in d.files:
                img[str(k)] = d[k].astype(np.float32)
        return cls({k: lbl[k] for k in lbl.files}, img)

    def has(self, gid: str) -> bool:
        return gid in self.img

    def predict_vec(self, vec: np.ndarray) -> dict | None:
        if vec is None:
            return None
        v = vec.astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-8)
        sims = self.T @ v  # [L]
        order = np.argsort(-sims)
        top = int(order[0])
        # macro skoru: her makro grubun en iyi prompt benzerliği
        macro_best: dict[str, float] = {}
        for i, m in enumerate(self.macro):
            if sims[i] > macro_best.get(m, -1):
                macro_best[m] = float(sims[i])
        ranked = sorted(macro_best.items(), key=lambda kv: -kv[1])
        macro_margin = ranked[0][1] - (ranked[1][1] if len(ranked) > 1 else 0.0)
        return {
            "macro": self.macro[top],
            "layer_role": self.layer_role[top],
            "category": self.category[top],
            "subcategory": self.subcategory[top],
            "score": float(sims[top]),
            "label_margin": float(sims[order[0]] - sims[order[1]]),
            "macro_score": ranked[0][1],
            "macro_margin": float(macro_margin),
            "macro_ranked": ranked,
        }

    def predict_id(self, gid: str) -> dict | None:
        vec = self.img.get(gid)
        return self.predict_vec(vec) if vec is not None else None


# layer_role -> kaba eksen (üst/alt/elbise/ayakkabı/aksesuar)
def macro_of_layer(layer_role: str | None) -> str:
    if layer_role in ("base", "mid", "outer", "top"):
        return "upper"
    if layer_role == "bottom":
        return "lower"
    if layer_role == "dress":
        return "dress"
    if layer_role == "footwear":
        return "footwear"
    if layer_role == "accessory":
        return "accessory"
    return "?"
