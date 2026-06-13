# Veri Kaynağı ve Lisans Kaydı (Data Provenance)

StylePops görsel envanter ve estetik uyumluluk hattı için kullanılan tüm veri kaynakları.

## Üretim gardırobu (uygulamada gösterilir)

| Kaynak | Lisans | Kullanım | Görsel saklama |
|--------|--------|----------|----------------|
| [Livostyle Open Data](https://github.com/arturayupov/womens-fashion-catalog-open-data) | **MIT** | Ticari + akademik | `data/assets/garments/` lokal mirror |
| [Fashion Product Images Small (44K)](https://huggingface.co/datasets/ashraq/fashion-product-images-small) | **MIT** | Filtreli takviye (SP*) | `data/assets/fashion_product/` |
| [fnauman — Clothing Dataset for Second-Hand Fashion](https://huggingface.co/datasets/fnauman/fashion-second-hand-front-only-rgb) | **CC-BY 4.0** | Kışlık dış giyim takviyesi (FN*) | `data/assets/garments/fnauman/` |
| Kullanıcı gardırobu (gelecek) | Kullanıcı mülkiyeti | Üretim | `data/assets/user/` |

**CC-BY 4.0 atıf (fnauman):** Nauman, F. (2024). *Clothing Dataset for Second-Hand Fashion* (Version 3) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13788681 — Data: Wargön Innovation AB, Myrorna AB; curation: RISE Research Institutes of Sweden AB.

**Kural:** Runtime'da dış URL'ye (hotlink) bağlanılmaz. `import_livostyle.py` her importta görselleri indirir ve `snapshot_date` yazar.

## Eğitim / estetik uyumluluk (görsel dağıtılmaz)

| Kaynak | Lisans | Kullanım |
|--------|--------|----------|
| [FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip) | Araştırma ağırlıklı | Pretrained embedding |
| [Fashion Product Images Small](https://huggingface.co/datasets/benitomartin/fashion-product-images-small-384x512) | MIT | İleride fine-tune (opsiyonel) |

## Kullanılmayan

| Kaynak | Neden |
|--------|-------|
| E-ticaret scrape | ToS + telif riski |
| DeepFashion | Sadece akademik, ticari yasak |

## Snapshot politikası

1. Haftalık sync: `python scripts/import_livostyle.py --refresh`
2. Kalkan ürünler: `active: false` — görsel silinmez
3. `data/visual/manifest.json` her importta güncellenir
