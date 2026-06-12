# StylePops — Proje Durumu

## Git durumu

**Görsel pipeline henüz GitHub'a push edilmedi** (scriptler, notebook, app yerel).

Push edilmesi gerekenler:
- `scripts/`, `app/`, `notebooks/StylePops_FashionCLIP_Training.ipynb`
- `data/visual/inventory_registry.json`, `manifest.json`, `garments_livostyle.json` (metadata)
- `PROJE_DURUMU.md`, `DATA_PROVENANCE.md`, `requirements-visual.txt`

**Push edilmeyecek** (`.gitignore`):
- `data/assets/` görselleri (~2GB+)
- `.env`, `outputs/`

## Veri (Mac'te hazır)

| Kaynak | Durum |
|--------|-------|
| Livostyle 2580 | ✅ |
| Fashion Product 44K | ✅ |
| LoRA eğitimi | ⏳ Colab'da yapılacak |
| Compatibility head | ⏳ LoRA sonrası Mac |

## Colab'da LoRA eğitimi (önerilen)

1. **Önce push:** Mac'te commit + `git push`
2. Colab: [StylePops_FashionCLIP_Training.ipynb](https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_FashionCLIP_Training.ipynb)
3. Runtime → **GPU T4**
4. Secrets → `HF_TOKEN`
5. Hücreleri çalıştır
6. `aesthetic_models.zip` indir → Mac `outputs/fashionclip_lora/`

Tek komut (Colab):
```bash
python scripts/colab_train_aesthetic.py --source hf --epochs 1 --batch-size 32 --zip
```

`--source hf` → 44K HuggingFace'ten iner, **Drive upload gerekmez**.

## Mac'te compatibility head (LoRA sonrası)

```bash
python scripts/train_compatibility_head.py
```

## Yerel LoRA (yavaş, önerilmez)

```bash
python scripts/colab_train_aesthetic.py --source local --epochs 1
```
