#!/usr/bin/env python3
"""StylePops görsel pipeline — tam ölçek: Livostyle + 44K MIT + eğitim."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def run_script(name: str, *args: str) -> int:
    cmd = [sys.executable, str(SCRIPTS / name), *args]
    print(f"\n{'='*60}\n▶ {' '.join(cmd)}\n{'='*60}")
    return subprocess.call(cmd, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="StylePops tam görsel pipeline")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--import-livostyle", action="store_true")
    parser.add_argument("--import-fp44k", action="store_true", help="Fashion Product 44K MIT")
    parser.add_argument("--combinations", action="store_true")
    parser.add_argument("--aesthetic-cache", action="store_true")
    parser.add_argument("--train-lora", action="store_true", help="FashionCLIP LoRA fine-tune")
    parser.add_argument("--train-compat", action="store_true", help="Compatibility head")
    parser.add_argument("--livostyle-sample", type=int, default=0, help="0=tüm katalog")
    parser.add_argument("--fp-limit", type=int, default=0, help="0=tüm 44K")
    parser.add_argument("--lora-samples", type=int, default=0, help="0=tüm FP korpus")
    parser.add_argument("--lora-epochs", type=int, default=1)
    args = parser.parse_args()

    if args.all:
        args.import_livostyle = True
        args.import_fp44k = True
        args.combinations = True
        args.aesthetic_cache = True
        args.train_lora = True
        args.train_compat = True

    if not any([
        args.import_livostyle, args.import_fp44k, args.combinations,
        args.aesthetic_cache, args.train_lora, args.train_compat,
    ]):
        parser.print_help()
        sys.exit(1)

    rc = 0

    if args.import_livostyle:
        liv_args = ["--target", str(args.livostyle_sample)]
        rc = run_script("import_livostyle.py", *liv_args) or rc

    if args.import_fp44k:
        fp_args = []
        if args.fp_limit > 0:
            fp_args.extend(["--limit", str(args.fp_limit)])
        fp_args.append("--resume")
        rc = run_script("import_fashion_product_images.py", *fp_args) or rc

    if args.combinations:
        rc = run_script(
            "generate_visual_combinations.py",
            "--per-scenario", "50", "--ab-pairs", "40", "--collages", "8",
        ) or rc

    if args.aesthetic_cache:
        sys.path.insert(0, str(SCRIPTS))
        from aesthetic_compatibility import load_garments, precompute_garment_embeddings
        import json

        garments = load_garments("production")
        embeddings = precompute_garment_embeddings(garments)
        out = ROOT / "data" / "visual" / "embeddings_cache.json"
        out.write_text(json.dumps(embeddings), encoding="utf-8")
        print(f"Embedding cache → {out}")

    if args.train_lora:
        lora_args = ["--epochs", str(args.lora_epochs)]
        if args.lora_samples > 0:
            lora_args.extend(["--max-samples", str(args.lora_samples)])
        rc = run_script("train_fashionclip_lora.py", *lora_args) or rc

    if args.train_compat:
        rc = run_script("train_compatibility_head.py") or rc

    if rc == 0:
        run_script("update_status.py")
        print("\n✓ Pipeline tamamlandı.")
        print("  streamlit run app/streamlit_app.py")
    sys.exit(rc)


if __name__ == "__main__":
    main()
