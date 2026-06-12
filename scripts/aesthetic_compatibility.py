#!/usr/bin/env python3
"""Estetik uyumluluk — FashionCLIP (LoRA fine-tune) + compatibility head + renk fallback."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

_FASHION_CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None
_clip_mode = "base"
_compat_bundle = None


def _lora_path() -> Path:
    reg_path = ROOT / "data" / "visual" / "inventory_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        return ROOT / reg["models"]["fashionclip_lora"]
    return ROOT / "outputs" / "fashionclip_lora"


def _compat_path() -> Path:
    reg_path = ROOT / "data" / "visual" / "inventory_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        return ROOT / reg["models"]["compatibility_head"]
    return ROOT / "outputs" / "compatibility_head_v1.joblib"


def _try_load_fashion_clip():
    global _FASHION_CLIP_AVAILABLE, _clip_model, _clip_processor, _clip_mode
    if _clip_model is not None:
        return _FASHION_CLIP_AVAILABLE
    try:
        from hf_env import ensure_hf_token
        ensure_hf_token()
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image

        _ = Image  # noqa: F841
        lora_dir = _lora_path()
        if lora_dir.exists() and (lora_dir / "adapter_config.json").exists():
            from peft import PeftModel

            base_id = "patrickjohncyh/fashion-clip"
            _clip_processor = CLIPProcessor.from_pretrained(str(lora_dir))
            base = CLIPModel.from_pretrained(base_id)
            _clip_model = PeftModel.from_pretrained(base, str(lora_dir))
            _clip_mode = "lora"
        else:
            model_id = "patrickjohncyh/fashion-clip"
            _clip_processor = CLIPProcessor.from_pretrained(model_id)
            _clip_model = CLIPModel.from_pretrained(model_id)
            _clip_mode = "base"
        _clip_model.eval()
        _FASHION_CLIP_AVAILABLE = True
    except Exception:
        _FASHION_CLIP_AVAILABLE = False
    return _FASHION_CLIP_AVAILABLE


def _load_compat_head():
    global _compat_bundle
    if _compat_bundle is not None:
        return _compat_bundle
    path = _compat_path()
    if not path.exists():
        return None
    try:
        import joblib
        _compat_bundle = joblib.load(path)
    except Exception:
        _compat_bundle = None
    return _compat_bundle


def load_garments(which: str = "production") -> dict[str, dict]:
    from inventory_loader import load_production_garments, load_training_garments

    if which == "training":
        return load_training_garments()
    return load_production_garments()


def color_harmony_score(piece_ids: list[str], garments: dict[str, dict]) -> float:
    labs = []
    for pid in piece_ids:
        g = garments.get(pid)
        if not g:
            continue
        c = g["color_lab"]
        labs.append([c["L"], c["a"], c["b"]])
    if len(labs) < 2:
        return 3.5
    arr = np.array(labs)
    std = float(np.std(arr, axis=0).mean())
    return float(max(1.0, min(5.0, 5.0 - std / 8.0)))


@lru_cache(maxsize=4096)
def _image_embedding_cached(garment_id: str, image_path: str, mtime: float, mode: str) -> tuple | None:
    del mtime, mode
    if not _try_load_fashion_clip():
        return None
    path = ROOT / image_path
    if not path.exists():
        return None
    try:
        import torch
        from PIL import Image

        img = Image.open(path).convert("RGB")
        inputs = _clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = _clip_model.get_image_features(**inputs)
            if isinstance(out, torch.Tensor):
                feats = out
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                feats = out.pooler_output
            else:
                return None
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return tuple(feats.squeeze().cpu().numpy().tolist())
    except Exception:
        return None


def garment_image_embedding(garment: dict) -> np.ndarray | None:
    image_path = garment.get("image_path")
    if not image_path:
        return None
    path = ROOT / image_path
    mtime = path.stat().st_mtime if path.exists() else 0.0
    cached = _image_embedding_cached(garment["id"], image_path, mtime, _clip_mode)
    if cached is None:
        return None
    return np.array(cached, dtype=np.float32)


def _outfit_feature(embs: list[np.ndarray]) -> np.ndarray | None:
    if len(embs) < 2:
        return None
    mat = np.stack(embs)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    sims = [float(np.dot(mat[i], mat[j])) for i in range(len(mat)) for j in range(i + 1, len(mat))]
    return np.concatenate([mean, std, [np.mean(sims), np.min(sims), np.max(sims)]])


def compatibility_head_score(piece_ids: list[str], garments: dict[str, dict]) -> float | None:
    bundle = _load_compat_head()
    if bundle is None:
        return None
    embs = []
    for pid in piece_ids:
        g = garments.get(pid)
        if not g:
            continue
        e = garment_image_embedding(g)
        if e is not None:
            embs.append(e)
    feat = _outfit_feature(embs)
    if feat is None:
        return None
    prob = float(bundle["model"].predict_proba(feat.reshape(1, -1))[0, 1])
    return round(1.0 + 4.0 * prob, 3)


def fashionclip_compatibility_score(piece_ids: list[str], garments: dict[str, dict]) -> float | None:
    embs = []
    for pid in piece_ids:
        g = garments.get(pid)
        if not g:
            continue
        emb = garment_image_embedding(g)
        if emb is not None:
            embs.append(emb)
    if len(embs) < 2:
        return None
    sims = [float(np.dot(embs[i], embs[j])) for i in range(len(embs)) for j in range(i + 1, len(embs))]
    mean_sim = float(np.mean(sims))
    score = 1.0 + 4.0 * max(0.0, min(1.0, (mean_sim - 0.45) / 0.40))
    return round(score, 3)


def aesthetic_compatibility_score(
    piece_ids: list[str],
    garments: dict[str, dict],
    *,
    fashionclip_weight: float = 0.5,
    compat_weight: float = 0.3,
) -> dict[str, float | str | None]:
    color = color_harmony_score(piece_ids, garments)
    clip = fashionclip_compatibility_score(piece_ids, garments)
    compat = compatibility_head_score(piece_ids, garments)

    parts = []
    weights = []
    if compat is not None:
        parts.append(compat)
        weights.append(compat_weight)
    if clip is not None:
        parts.append(clip)
        weights.append(fashionclip_weight)
    parts.append(color)
    weights.append(max(0.1, 1.0 - sum(weights)))

    if clip is None and compat is None:
        return {
            "aesthetic_score": round(color, 3),
            "fashionclip_score": None,
            "compatibility_head_score": None,
            "color_score": round(color, 3),
            "scorer": "color_fallback",
            "clip_mode": _clip_mode,
        }

    wsum = sum(weights)
    combined = sum(p * w for p, w in zip(parts, weights)) / wsum
    scorer = f"{_clip_mode}"
    if compat is not None:
        scorer += "+compat_head"
    if clip is not None:
        scorer += "+fashionclip"
    scorer += "+color"

    return {
        "aesthetic_score": round(combined, 3),
        "fashionclip_score": clip,
        "compatibility_head_score": compat,
        "color_score": round(color, 3),
        "scorer": scorer,
        "clip_mode": _clip_mode,
    }


def make_aesthetic_fn(garments: dict[str, dict]) -> Callable[[list[str]], float]:
    def fn(piece_ids: list[str]) -> float:
        return float(aesthetic_compatibility_score(piece_ids, garments)["aesthetic_score"])

    return fn


def precompute_garment_embeddings(garments: dict[str, dict]) -> dict[str, list[float] | None]:
    if not _try_load_fashion_clip():
        print("FashionCLIP yüklenemedi — requirements-visual.txt kurun.")
        return {}
    result = {}
    for i, (gid, g) in enumerate(garments.items(), 1):
        emb = garment_image_embedding(g)
        result[gid] = emb.tolist() if emb is not None else None
        if i % 500 == 0:
            print(f"  … {i}/{len(garments)} embedding")
    ok = sum(1 for v in result.values() if v is not None)
    print(f"Embedding önbellek ({_clip_mode}): {ok}/{len(garments)} parça")
    return result
