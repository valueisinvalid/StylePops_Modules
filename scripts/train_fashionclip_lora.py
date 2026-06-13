#!/usr/bin/env python3
"""
FashionCLIP LoRA fine-tune — Fashion Product Images 44K (MIT).
Colab: --source hf (bellek dostu, tüm dataset'i listeye yüklemez).
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


def load_hf_dataset(max_samples: int):
    from datasets import load_dataset

    if max_samples:
        split = f"train[:{max_samples}]"
    else:
        split = "train"
    print(f"HF dataset: {HF_DATASET_ID} ({split})")
    return load_dataset(HF_DATASET_ID, split=split)


def build_local_pairs(max_samples: int, seed: int) -> list[tuple]:
    garments = load_training_garments()
    if not garments:
        return []
    rng = random.Random(seed)
    items = [g for g in garments.values() if (ROOT / g["image_path"]).exists()]
    rng.shuffle(items)
    limit = max_samples or len(items)
    pairs = []
    for g in items[:limit]:
        text = g.get("name") or g.get("description", "")
        pairs.append(("local", ROOT / g["image_path"], text))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("local", "hf"), default="local")
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
        if ensure_hf_token():
            print("HF token yüklendi")
        else:
            print("Uyarı: HF_TOKEN yok — Colab Secrets veya .env kullanın")
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        from transformers import CLIPModel, CLIPProcessor
        from peft import LoraConfig, get_peft_model
        from PIL import Image
    except ImportError:
        print("pip install torch transformers peft datasets pillow", file=sys.stderr)
        sys.exit(1)

    hf_ds = None
    local_pairs = None

    if args.source == "hf":
        hf_ds = load_hf_dataset(args.max_samples)
        n_samples = len(hf_ds)
        print(f"Kaynak: hf | Örnek sayısı: {n_samples}")
    else:
        local_pairs = build_local_pairs(args.max_samples, args.seed)
        if not local_pairs:
            print("Önce: python scripts/import_fashion_product_images.py", file=sys.stderr)
            sys.exit(1)
        n_samples = len(local_pairs)
        print(f"Kaynak: local | Örnek sayısı: {n_samples}")

    class HFPairDS(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            row = self.dataset[idx]
            return row["image"], row.get("productDisplayName") or ""

    class LocalPairDS(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            _, path, text = self.data[idx]
            return path, text

    model_id = "patrickjohncyh/fashion-clip"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Cihaz: {device}")
    if device.type != "cuda":
        print("Uyarı: GPU yok — Colab Runtime → T4 seçin")
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def collate_pil_batch(batch):
        images, texts = zip(*batch)
        return list(images), list(texts)

    def collate_local_batch(batch):
        paths, texts = zip(*batch)
        return list(paths), list(texts)

    if args.source == "hf":
        dl = DataLoader(
            HFPairDS(hf_ds), batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_pil_batch,
        )
    else:
        dl = DataLoader(
            LocalPairDS(local_pairs), batch_size=args.batch_size, shuffle=True,
            num_workers=0, collate_fn=collate_local_batch,
        )

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in dl:
            if args.source == "hf":
                pil_images, texts = batch
                images = [img.convert("RGB") for img in pil_images]
            else:
                paths, texts = batch
                images = [Image.open(p).convert("RGB") for p in paths]
                texts = list(texts)

            inputs = processor(
                text=list(texts), images=images, return_tensors="pt",
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
                print(f"  epoch {epoch + 1} batch {n_batches}/{len(dl)} loss={loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{args.epochs} loss={total_loss / max(n_batches, 1):.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    processor.save_pretrained(OUT_DIR)
    meta = {
        "base_model": model_id,
        "source": args.source,
        "samples": n_samples,
        "epochs": args.epochs,
    }
    (OUT_DIR / "training_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"LoRA kaydedildi → {OUT_DIR}")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
