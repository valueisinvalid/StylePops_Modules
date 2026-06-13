# StylePops — Proje Durumu

## Colab — FashionCLIP LoRA (GitHub)

**Notebook:** https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_FashionCLIP_Training.ipynb

1. Runtime → **T4 GPU**
2. Colab **Secrets** → `HF_TOKEN` (tokenını kendin yaz)
3. Hücreleri sırayla çalıştır
4. Önce 3000 örnek test → OK ise tam 44K
5. `fashionclip_lora.zip` indir → Mac `outputs/fashionclip_lora/`
6. Mac: `python scripts/train_compatibility_head.py`

## Veri (Mac'te, git dışı)

| Kaynak | Boyut | Git |
|--------|-------|-----|
| Livostyle 2580 | 3MB JSON + görseller | JSON ✅ görseller ❌ |
| Fashion Product 44K | 41MB JSON + 2GB görseller | ❌ Colab HF'den çeker |

## Mac'te import (bir kez yapıldı)

```bash
python scripts/import_livostyle.py
python scripts/import_fashion_product_images.py --resume
```
