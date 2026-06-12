"""Hugging Face token — .env veya ortam değişkeninden yükler. Token repoya yazılmaz."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ensure_hf_token() -> bool:
    """HF_TOKEN / HUGGING_FACE_HUB_TOKEN ayarla. Token varsa True."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True

    env_path = ROOT / ".env"
    if not env_path.exists():
        return False

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("HF_TOKEN="):
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val:
                os.environ["HF_TOKEN"] = val
                os.environ["HUGGING_FACE_HUB_TOKEN"] = val
                return True
    return False
