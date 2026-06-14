#!/usr/bin/env python3
"""FashionCLIP metin etiketi embedding'lerini bir kez hesaplayıp kaydet.

`visual_classify.py` bu embedding'leri kullanarak (torch'a ihtiyaç duymadan)
önbellekteki görsel embedding'lerle kosinüs benzerliği üzerinden her parçanın
GÖRSEL kategorisini çıkarır. Böylece tüm gardırop yanlış-etiket taraması anında
yapılır.

Çıktı: data/visual/clip_text_labels.npz
    prompts (str[]), layer_role (str[]), category (str[]),
    subcategory (str[]), macro (str[]), embeddings (float32[N,512])

macro: kaba slot — üst/alt/elbise/ayakkabı/aksesuar ekseninde KABA hata yakalamak
için. (upper = base/mid/outer/top hepsi 'upper'.)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "visual" / "clip_text_labels.npz"
MODEL_ID = "patrickjohncyh/fashion-clip"

# (prompt, layer_role, category, subcategory, macro)
# Her sınıf için birden çok prompt → daha sağlam. En iyi eşleşen prompt'un
# (layer_role, category, subcategory) bilgisi düzeltme için kullanılır.
LABELS: list[tuple[str, str, str, str, str]] = [
    # ---- ALT (bottom) ----
    ("a photo of a pair of trousers", "bottom", "bottom", "trousers", "lower"),
    ("a photo of a pair of pants", "bottom", "bottom", "trousers", "lower"),
    ("a photo of a pair of chinos", "bottom", "bottom", "trousers", "lower"),
    ("a photo of a pair of jeans", "bottom", "bottom", "jeans", "lower"),
    ("a photo of a pair of leggings", "bottom", "bottom", "leggings", "lower"),
    ("a photo of a pair of shorts", "bottom", "bottom", "shorts", "lower"),
    ("a photo of a skirt", "bottom", "bottom", "skirt", "lower"),
    # ---- ÜST / BAZ (base) ----
    ("a photo of a t-shirt", "base", "top", "tshirt", "upper"),
    ("a photo of a shirt", "mid", "top", "shirt", "upper"),
    ("a photo of a blouse", "mid", "top", "blouse", "upper"),
    ("a photo of a tank top", "base", "top", "tank_top", "upper"),
    ("a photo of a polo shirt", "base", "top", "polo", "upper"),
    # ---- ARA KATMAN (mid) ----
    ("a photo of a sweater", "mid", "top", "sweater", "upper"),
    ("a photo of a knit pullover", "mid", "top", "sweater", "upper"),
    ("a photo of a hoodie", "mid", "top", "hoodie", "upper"),
    ("a photo of a sweatshirt", "mid", "top", "sweatshirt", "upper"),
    ("a photo of a cardigan", "mid", "top", "cardigan", "upper"),
    ("a photo of a sleeveless sweater vest", "mid", "top", "vest", "upper"),
    # ---- DIŞ GİYİM (outer) ----
    ("a photo of a jacket", "outer", "outerwear", "jacket", "upper"),
    ("a photo of a denim jacket", "outer", "outerwear", "jacket", "upper"),
    ("a photo of a blazer", "outer", "outerwear", "blazer", "upper"),
    ("a photo of a coat", "outer", "outerwear", "coat", "upper"),
    ("a photo of a winter puffer coat", "outer", "outerwear", "padded_coat", "upper"),
    ("a photo of a raincoat", "outer", "outerwear", "raincoat", "upper"),
    # ---- ELBİSE (dress) ----
    ("a photo of a dress", "dress", "dress", "dress", "dress"),
    ("a photo of a jumpsuit", "dress", "dress", "jumpsuit", "dress"),
    # ---- AYAKKABI (footwear) ----
    ("a photo of a pair of boots", "footwear", "footwear", "boots", "footwear"),
    ("a photo of a pair of sneakers", "footwear", "footwear", "sneakers", "footwear"),
    ("a photo of a pair of sandals", "footwear", "footwear", "sandals", "footwear"),
    ("a photo of a pair of high heels", "footwear", "footwear", "heels", "footwear"),
    ("a photo of a pair of loafers", "footwear", "footwear", "loafers", "footwear"),
    ("a photo of a pair of flat shoes", "footwear", "footwear", "flats", "footwear"),
    # ---- AKSESUAR (accessory) ----
    ("a photo of a scarf", "accessory", "accessory", "scarf", "accessory"),
    ("a photo of a hat", "accessory", "accessory", "hat", "accessory"),
    ("a photo of a handbag", "accessory", "accessory", "bag", "accessory"),
    ("a photo of a belt", "accessory", "accessory", "belt", "accessory"),
]


def _feats(out, torch):
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("text_embeds", "image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return None


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    import torch
    import torch.nn.functional as F
    from transformers import CLIPModel, CLIPProcessor

    dev = _device(torch)
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval().to(dev)
    prompts = [r[0] for r in LABELS]

    @torch.no_grad()
    def embed(texts):
        inp = {k: v.to(dev) for k, v in proc(text=texts, return_tensors="pt", padding=True).items()}
        return F.normalize(_feats(model.get_text_features(**inp), torch), dim=-1)

    emb = embed(prompts).cpu().numpy().astype(np.float32)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        prompts=np.array(prompts, dtype=object),
        layer_role=np.array([r[1] for r in LABELS], dtype=object),
        category=np.array([r[2] for r in LABELS], dtype=object),
        subcategory=np.array([r[3] for r in LABELS], dtype=object),
        macro=np.array([r[4] for r in LABELS], dtype=object),
        embeddings=emb,
    )
    print(f"Kaydedildi → {OUT} ({len(prompts)} etiket, dim {emb.shape[1]}, cihaz {dev})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
