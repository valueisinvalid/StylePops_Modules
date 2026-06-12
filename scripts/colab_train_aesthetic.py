#!/usr/bin/env python3
"""
Colab estetik eğitim giriş noktası — LoRA + compatibility head.

Kullanım (Colab GPU):
  export HF_TOKEN=hf_...
  python scripts/colab_train_aesthetic.py

Yerel (44K zaten indirildiyse):
  python scripts/colab_train_aesthetic.py --source local
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUT_LORA = ROOT / "outputs" / "fashionclip_lora"
OUT_COMPAT = ROOT / "outputs" / "compatibility_head_v1.joblib"
ZIP_OUT = ROOT / "outputs" / "aesthetic_models.zip"


def run(name: str, *args: str) -> int:
    cmd = [sys.executable, str(SCRIPTS / name), *args]
    print(f"\n{'='*60}\n▶ {' '.join(cmd)}\n{'='*60}")
    return subprocess.call(cmd, cwd=ROOT)


def zip_outputs() -> None:
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        if OUT_LORA.exists():
            for p in OUT_LORA.rglob("*"):
                if p.is_file():
                    zf.write(p, p.relative_to(ROOT))
        if OUT_COMPAT.exists():
            zf.write(OUT_COMPAT, OUT_COMPAT.relative_to(ROOT))
    print(f"Model paketi → {ZIP_OUT}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab / yerel estetik eğitim")
    parser.add_argument(
        "--source",
        choices=("hf", "local"),
        default="hf",
        help="hf = HuggingFace 44K (Colab, 2GB upload gerekmez); local = Mac import",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="0 = tam 44K")
    parser.add_argument("--batch-size", type=int, default=32, help="Colab T4 için 32")
    parser.add_argument("--skip-lora", action="store_true")
    parser.add_argument("--skip-compat", action="store_true")
    parser.add_argument("--compat-embedding-sample", type=int, default=8000)
    parser.add_argument("--zip", action="store_true", help="outputs/aesthetic_models.zip oluştur")
    args = parser.parse_args()

    if not os.environ.get("HF_TOKEN"):
        env_file = ROOT / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    os.environ["HF_TOKEN"] = line.split("=", 1)[1].strip()

    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Cihaz: {device}")
    if device == "cpu":
        print("Uyarı: GPU yok — Colab'da Runtime → GPU seçin.")

    rc = 0
    if not args.skip_lora:
        lora_args = [
            "--source", args.source,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--log-every", "100",
        ]
        if args.max_samples > 0:
            lora_args.extend(["--max-samples", str(args.max_samples)])
        rc = run("train_fashionclip_lora.py", *lora_args) or rc

    if not args.skip_compat:
        if args.source == "hf":
            print(
                "\nCompatibility head için yerel 44K korpus gerekir.\n"
                "Colab'da atlanıyor — Mac'te çalıştırın:\n"
                "  python scripts/train_compatibility_head.py\n"
            )
        else:
            rc = run(
                "train_compatibility_head.py",
                "--embedding-sample", str(args.compat_embedding_sample),
            ) or rc

    if args.zip or (rc == 0 and OUT_LORA.exists()):
        zip_outputs()

    if rc == 0:
        print("\n✓ Eğitim tamam.")
        if OUT_LORA.exists():
            print(f"  LoRA: {OUT_LORA}")
            print("  Mac'e kopyala: outputs/fashionclip_lora/")
        if OUT_COMPAT.exists():
            print(f"  Compat: {OUT_COMPAT}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
