# StylePops — Google Colab Kullanım Rehberi (GitHub)

## En Hızlı Yol

1. Bu linke tıkla (repo push edildikten sonra aktif olur):

   **https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_Bootstrap_Pipeline.ipynb**

2. **Runtime → Change runtime type → GPU (T4)**
3. **Runtime → Run all** (veya hücre hücre)

İlk hücre repo'yu otomatik clone eder — Drive veya zip gerekmez.

---

## Notebook Akışı

| Sıra | Hücre | Ne yapar |
|------|-------|----------|
| 1 | `pip install` | Bağımlılıklar |
| 2 | `git clone` | GitHub'dan proje indirilir |
| 3 | Veri yükleme | 200 parça + 200 kombin |
| 4–7 | İP-1 / İP-2 | Özellik + termal model |
| 8 | LightGBM | Etiket varsa eğitim |
| 9–10 | İP-3 + Demo | 4 mevsim önerisi |
| 11–12 | Değerlendirme + kayıt | CSV export |

---

## Etiketleme (GitHub üzerinden)

1. GitHub'da `data/bootstrap/combinations_200.csv` dosyasını düzenle
2. `aesthetic_score` (1–5) ve `thermal_score` (1–3) doldur
3. Commit et
4. Colab'da notebook'u yeniden çalıştır — `git pull` güncel etiketleri çeker

En az 20 etiketli satır → LightGBM devreye girer.

---

## Kendi Fork'un

```python
REPO_URL = "https://github.com/KULLANICI_ADIN/StylePops_Modules.git"
```

---

## Mac'te Güncelleme → GitHub → Colab

```bash
cd /Users/o_7/StylePops_Modules
# değişiklik yap
git add .
git commit -m "açıklama"
git push
```

Colab'da clone hücresi `git pull` ile güncellemeyi alır.

---

## Veriyi Yeniden Üretme

```bash
python3 scripts/generate_bootstrap_data.py
git add data/bootstrap/
git commit -m "regenerate bootstrap data"
git push
```

---

## Sık Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `git clone` hata | Repo public mi? URL doğru mu? |
| Eski veri | Clone hücresini tekrar çalıştır (`git pull`) |
| SBERT yavaş | GPU runtime seç |
| LightGBM atlandı | CSV'de min 20 etiket |
