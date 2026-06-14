#!/usr/bin/env python3
"""Livostyle parçalarını ORİJİNAL görsellerinden FashionCLIP ile doğrula.

Amaç (kullanıcı isteği):
  - Kategorilemeyi fotoğraftan tanıyan modelle (FashionCLIP, MIT) çapraz kontrol et.
  - Yerel LV görselleri 200×200 merkez-kırpık; sınıflandırma ORİJİNALDEN yapılmalı.
  - Modelin algıladığını ürün adı/etiketi ile karşılaştırıp gerçeği belirle.

Yapılanlar:
  1. Her LV için image_source_url'den orijinali indir (originals/ önbellek).
  2. FashionCLIP zero-shot → görsel kategori + skor.
  3. İsim-tabanlı etiketle karşılaştır, uyuşmazlıkları rapor et.
  4. LV embedding'lerini ORİJİNALDEN yeniden hesaplayıp clip_embeddings.npz'i güncelle
     (estetik skorlama da kırpık değil tam görselden gelsin).

Çıktı:
  - data/assets/garments/originals/LV*.jpg
  - data/visual/lv_visual_verify.json (rapor)
  - data/visual/clip_embeddings.npz (LV girdileri güncellenir)
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from garment_name_classifier import classify as classify_name, layer_group

PROD = ROOT / "data" / "visual" / "garments_production.json"
ORIG_DIR = ROOT / "data" / "assets" / "garments" / "originals"
CACHE = ROOT / "data" / "visual" / "clip_embeddings.npz"
REPORT = ROOT / "data" / "visual" / "lv_visual_verify.json"
MODEL_ID = "patrickjohncyh/fashion-clip"

# zero-shot etiketleri → katman grubu
LABELS = [
    ("a photo of a t-shirt", "top"), ("a photo of a shirt", "top"),
    ("a photo of a blouse", "top"), ("a photo of a sweater", "top"),
    ("a photo of a cardigan", "top"), ("a photo of a hoodie", "top"),
    ("a photo of a tank top", "top"),
    ("a photo of trousers", "bottom"), ("a photo of jeans", "bottom"),
    ("a photo of shorts", "bottom"), ("a photo of a skirt", "bottom"),
    ("a photo of a dress", "dress"), ("a photo of a jumpsuit", "dress"),
    ("a photo of a coat", "outer"), ("a photo of a jacket", "outer"),
    ("a photo of a blazer", "outer"),
    ("a photo of shoes", "footwear"), ("a photo of boots", "footwear"),
    ("a photo of sandals", "footwear"), ("a photo of high heels", "footwear"),
]


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _feats(out, torch):
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    return None


def fetch_original(g: dict) -> Path | None:
    gid = g["id"]
    dst = ORIG_DIR / f"{gid}.jpg"
    if dst.exists():
        return dst
    url = g.get("image_source_url")
    if not url:
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = urllib.request.urlopen(req, timeout=25).read()
        from PIL import Image
        im = Image.open(io.BytesIO(data)).convert("RGB")
        im.save(dst, "JPEG", quality=90)
        return dst
    except Exception:
        return None


def main() -> int:
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    ORIG_DIR.mkdir(parents=True, exist_ok=True)
    garments = json.loads(PROD.read_text(encoding="utf-8"))["garments"]
    lv = [g for g in garments if g["id"].startswith("LV")]
    print(f"LV parça: {len(lv)} — orijinaller indiriliyor + FashionCLIP doğrulama")

    dev = _device(torch)
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval().to(dev)

    @torch.no_grad()
    def embed_text(texts):
        inp = {k: v.to(dev) for k, v in proc(text=texts, return_tensors="pt", padding=True).items()}
        return F.normalize(_feats(model.get_text_features(**inp), torch), dim=-1)

    @torch.no_grad()
    def embed_image(img):
        inp = {k: v.to(dev) for k, v in proc(images=img, return_tensors="pt").items()}
        return F.normalize(_feats(model.get_image_features(**inp), torch), dim=-1)

    T = embed_text([t for t, _ in LABELS])
    groups = [g for _, g in LABELS]

    cache = dict(np.load(CACHE)) if CACHE.exists() else {}
    report = {"agree": 0, "disagree": 0, "no_original": 0, "items": []}
    t0 = time.time()
    for i, g in enumerate(lv, 1):
        gid = g["id"]
        path = fetch_original(g)
        used_original = path is not None
        if path is None:
            report["no_original"] += 1
            local = ROOT / g.get("image_path", "")
            if not local.exists():
                continue
            path = local
        try:
            img = Image.open(path).convert("RGB")
            emb = embed_image(img)
            cache[gid] = emb.squeeze().cpu().numpy().astype(np.float32)
            sims = (emb @ T.T)[0]
            order = sims.argsort(descending=True)
            top = int(order[0])
            vlabel = LABELS[top][0].replace("a photo of ", "")
            vgroup = groups[top]
            margin = float(sims[order[0]] - sims[order[1]])
        except Exception:
            continue

        name_group = layer_group(g.get("layer_role"))
        agree = (vgroup == name_group)
        report["agree" if agree else "disagree"] += 1
        if not agree:
            report["items"].append({
                "id": gid, "name": g["name"][:70],
                "label": name_group, "subcategory": g.get("subcategory"),
                "visual": vgroup, "visual_label": vlabel,
                "score": round(float(sims[top]), 3), "margin": round(margin, 3),
                "original": used_original,
            })
        if i % 200 == 0:
            print(f"  … {i}/{len(lv)} ({time.time()-t0:.0f}s) uyum {report['agree']} fark {report['disagree']}")

    np.savez_compressed(CACHE, **cache)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    total = report["agree"] + report["disagree"]
    print(f"\nTamam ({time.time()-t0:.0f}s). Uyum {report['agree']}/{total} "
          f"({100*report['agree']//max(1,total)}%), orijinali yok {report['no_original']}")
    print(f"Rapor → {REPORT} · Cache güncellendi → {CACHE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
