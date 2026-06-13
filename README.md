# StylePops Modules

Çok kriterli giyim kombinasyonu optimizasyon modeli — bootstrap + **görsel envanter hattı**.

## Görsel envanter (yerel — önerilen)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-visual.txt   # FashionCLIP için

python scripts/run_visual_pipeline.py --all
streamlit run app/streamlit_app.py
```

Takip panosu: `PROJE_DURUMU.md` · Veri lisansları: `DATA_PROVENANCE.md`

## Google Colab (bootstrap İP-1→İP-3)

1. [Bootstrap notebook](https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_Bootstrap_Pipeline.ipynb)
2. **Runtime → GPU** → hücreleri sırayla çalıştır

## Colab — FashionCLIP LoRA eğitimi

1. [FashionCLIP Training notebook](https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_FashionCLIP_Training.ipynb)
2. **Runtime → T4 GPU**
3. Colab **Secrets** → `HF_TOKEN`
4. Hücreleri sırayla çalıştır → `fashionclip_lora.zip` indir

Detay: [PROJE_DURUMU.md](PROJE_DURUMU.md)

## İçerik

| Klasör | Açıklama |
|--------|----------|
| `data/bootstrap/` | 200 sentetik parça + kombin |
| `data/visual/` | 300 Livostyle görsel parça + kombin CSV |
| `data/assets/` | Mirror görseller (lokal, gitignore) |
| `scripts/` | import, estetik, pipeline |
| `app/` | Streamlit görsel arayüz |
