#!/usr/bin/env python3
"""
FashionCLIP LoRA fine-tune — Fashion Product Images 44K (MIT).
(image, productDisplayName) kontrastif hizalama; estetik embedding kalitesini artırır.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from inventory_loader import load_training_garments

OUT_DIR = ROOT / "outputs" / "fashionclip_lora"
HF_DATASET_ID = "benitomartin/fashion-product-images-small-384x512"


def iter_training_pairs_local(garments: dict, max_samples: int, seed: int):
    rng = random.Random(seed)
    items = [g for g in garments.values() if (ROOT / g["image_path"]).exists()]
    rng.shuffle(items)
    limit = max_samples or len(items)
    for g in items[:limit]:
        text = g.get("name") or g.get("description", "")
        yield ("local", ROOT / g["image_path"], text)


def iter_training_pairs_hf(max_samples: int, seed: int):
    from datasets import load_dataset

    split = f"train[:{max_samples}]" if max_samples else "train"
    print(f"HF dataset indiriliyor/önbellekten yükleniyor: {HF_DATASET_ID} ({split})")
    ds = load_dataset(HF_DATASET_ID, split=split)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    for i in indices:
        row = ds[i]
        text = row.get("productDisplayName") or ""
        yield ("hf", row["image"], text)


def build_pair_list(source: str, max_samples: int, seed: int) -> list[tuple]:
    if source == "hf":
        return list(iter_training_pairs_hf(max_samples, seed))
    garments = load_training_garments()
    if not garments:
        return []
    return list(iter_training_pairs_local(garments, max_samples, seed))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=("local", "hf"),
        default="local",
        help="local = data/assets/fashion_product; hf = HuggingFace (Colab için)",
    )
    parser.add_argument("--max-samples", type=int, default=0, help="0 = tüm korpus")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    try:
        from hf_env import ensure_hf_token
        ensure_hf_token()
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import CLIPModel, CLIPProcessor
        from peft import LoraConfig, get_peft_model
        from PIL import Image
    except ImportError:
        print("pip install -r requirements-visual.txt (torch, transformers, peft)", file=sys.stderr)
        sys.exit(1)

    garments = load_training_garments() if args.source == "local" else {}
    if args.source == "local" and not garments:
        print("Önce: python scripts/import_fashion_product_images.py", file=sys.stderr)
        print("veya Colab: --source hf", file=sys.stderr)
        sys.exit(1)

    max_n = args.max_samples or (len(garments) if args.source == "local" else 44072)
    pairs = build_pair_list(args.source, max_n, args.seed)
    if not pairs:
        print("Eğitim örneği bulunamadı.", file=sys.stderr)
        sys.exit(1)
    print(f"Kaynak: {args.source} | Eğitim örnekleri: {len(pairs)}")

    class PairDS(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            kind, img_ref, text = self.data[idx]
            return kind, img_ref, text

    model_id = "patrickjohncyh/fashion-clip"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Cihaz: {device}")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dl = DataLoader(PairDS(pairs), batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in dl:
            images = []
            texts = []
            for kind, img_ref, text in batch:
                if kind == "hf":
                    images.append(img_ref.convert("RGB"))
                else:
                    images.append(Image.open(img_ref).convert("RGB"))
                texts.append(text)
            inputs = processor(
                text=texts, images=images, return_tensors="pt",
                padding=True, truncation=True,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            img_e = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_e = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * img_e @ txt_e.T
            labels = torch.arange(logits.size(0), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
            if n_batches % args.log_every == 0:
                print(f"  epoch {epoch + 1} batch {n_batches} loss={loss.item():.4f}")
        print(f"Epoch {epoch + 1}/{args.epochs} loss={total_loss / max(n_batches, 1):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    meta = {
        "base_model": model_id,
        "source": args.source,
        "samples": len(pairs),
        "epochs": args.epochs,
        "license": "MIT training data",
    }
    (OUT_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"LoRA kaydedildi → {OUT_DIR}")


if __name__ == "__main__":
    main()
