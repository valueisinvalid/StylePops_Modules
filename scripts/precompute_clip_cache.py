#!/usr/bin/env python3
"""Tüm üretim gardırobu için FashionCLIP görsel embedding önbelleğini hesaplar.

Çıktı: data/visual/clip_embeddings.npz  (gid -> 512-boyut vektör)

Bu önbellek sayesinde kombin üretimi modeli yüklemeden FashionCLIP skoru
kullanabilir; Colab/torchao bağımlılığı ortadan kalkar. Mac (MPS/CPU) üzerinde
~1 dakikada biter, torchao gerektirmez (base patrickjohncyh/fashion-clip).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inventory_loader import load_production_garments

OUT_PATH = ROOT / "data" / "visual" / "clip_embeddings.npz"
MODEL_ID = "patrickjohncyh/fashion-clip"


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _features(out, torch):
    if isinstance(out, torch.Tensor):
        return out
    if getattr(out, "image_embeds", None) is not None:
        return out.image_embeds
    if getattr(out, "pooler_output", None) is not None:
        return out.pooler_output
    return None


def main() -> int:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    garments = load_production_garments()
    if not garments:
        print("HATA: üretim gardırobu boş", file=sys.stderr)
        return 1

    dev = _device(torch)
    print(f"FashionCLIP yükleniyor ({MODEL_ID}, {dev})…")
    t0 = time.time()
    proc = CLIPProcessor.from_pretrained(MODEL_ID)
    model = CLIPModel.from_pretrained(MODEL_ID).eval().to(dev)
    print(f"  model hazır {time.time() - t0:.1f}s")

    embeddings: dict[str, np.ndarray] = {}
    missing = 0
    t1 = time.time()
    items = list(garments.items())
    for i, (gid, g) in enumerate(items, 1):
        rel = g.get("image_path", "")
        path = ROOT / rel
        if not rel or not path.exists():
            missing += 1
            continue
        try:
            img = Image.open(path).convert("RGB")
            inp = proc(images=img, return_tensors="pt")
            inp = {k: v.to(dev) for k, v in inp.items()}
            with torch.no_grad():
                feats = _features(model.get_image_features(**inp), torch)
                if feats is None:
                    missing += 1
                    continue
                feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings[gid] = feats.squeeze().cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f"  {gid} atlandı: {exc}")
            missing += 1
        if i % 500 == 0:
            print(f"  … {i}/{len(items)} ({time.time() - t1:.0f}s)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **embeddings)
    mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(
        f"Önbellek kaydedildi → {OUT_PATH} "
        f"({len(embeddings)} parça, eksik {missing}, {mb:.1f} MB, "
        f"toplam {time.time() - t0:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
