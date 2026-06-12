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

1. [Colab notebook](https://colab.research.google.com/github/valueisinvalid/StylePops_Modules/blob/main/notebooks/StylePops_Bootstrap_Pipeline.ipynb)
2. **Runtime → GPU** → hücreleri sırayla çalıştır

Detay: [COLAB_KULLANIM.md](COLAB_KULLANIM.md)

## İçerik

| Klasör | Açıklama |
|--------|----------|
| `data/bootstrap/` | 200 sentetik parça + kombin |
| `data/visual/` | 300 Livostyle görsel parça + kombin CSV |
| `data/assets/` | Mirror görseller (lokal, gitignore) |
| `scripts/` | import, estetik, pipeline |
| `app/` | Streamlit görsel arayüz |
