# Veri Kaynağı ve Lisans Kaydı (Data Provenance)

StylePops görsel envanter ve estetik uyumluluk hattı için kullanılan tüm veri kaynakları.

## Üretim gardırobu (uygulamada gösterilir)

| Kaynak | Lisans | Kullanım | Görsel saklama |
|--------|--------|----------|----------------|
| [Livostyle Open Data](https://github.com/arturayupov/womens-fashion-catalog-open-data) | **MIT** | Ticari + akademik | `data/assets/garments/` lokal mirror |
| Kullanıcı gardırobu (gelecek) | Kullanıcı mülkiyeti | Üretim | `data/assets/user/` |

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
