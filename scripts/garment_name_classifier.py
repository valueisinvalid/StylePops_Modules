#!/usr/bin/env python3
"""Ürün adı + açıklamasından yüksek-isabetli kıyafet kategorisi çıkarımı.

Bazı kaynaklarda (özellikle Livostyle) kayıtlı category/subcategory/layer_role
alanları karışık/yanlış; ürün ADI ise güvenilir ("Wooden Clog Mule", "Cowboy
Boots" → ayakkabı). Bu modül adı/açıklamayı sıralı, yüksek-kesinlikli regex
kurallarıyla çözer ve (layer_role, category, subcategory) döndürür.

classify(name, desc) -> dict | None
    None  → addan net bir sinyal çıkmadı (mevcut etiketi koru / görsele bak)
    dict  → {layer_role, category, subcategory, signal} ; signal: eşleşen anahtar
"""

from __future__ import annotations

import re

# (subcategory, layer_role, category, regex) — SIRA ÖNEMLİ: ilk eşleşen kazanır.
# Daha spesifik / karışması zor olanlar (ayakkabı, aksesuar) en üstte.
_RULES: list[tuple[str, str, str, str]] = [
    # --- AYAKKABI (footwear) ---
    ("boots",       "footwear", "footwear", r"\b(boots?|booties|bootie|combat boot|chelsea boot|ankle boot)\b"),
    ("sandals",     "footwear", "footwear", r"\b(sandals?|slides?|flip[\s-]?flops?)\b"),
    ("sneakers",    "footwear", "footwear", r"\b(sneakers?|trainers?|kicks)\b"),
    ("loafers",     "footwear", "footwear", r"\b(loafers?|moccasins?|driving shoe)\b"),
    ("derby",       "footwear", "footwear", r"\b(oxfords?|derby|derbies|brogues?)\b"),
    ("heels",       "footwear", "footwear", r"\b(heels?|stilettos?|pumps?|wedges?)\b"),
    ("flats",       "footwear", "footwear", r"\b(ballet flats?|flats?|espadrilles?)\b"),
    ("clogs",       "footwear", "footwear", r"\b(clogs?|mules?|crocs?)\b"),
    ("slippers",    "footwear", "footwear", r"\b(slippers?)\b"),
    ("footwear",    "footwear", "footwear", r"\b(footwear|shoes?)\b"),

    # --- AKSESUAR / GİYİM-DIŞI (combo dışı) ---
    ("belt",        "accessory", "accessory", r"\b(belts?)\b"),
    ("bag",         "accessory", "accessory", r"\b(handbags?|bags?|totes?|purses?|clutch(?:es)?|backpacks?|satchel)\b"),
    ("hat",         "accessory", "accessory", r"\b(hats?|caps?|beanies?|berets?)\b"),
    ("scarf",       "accessory", "accessory", r"\b(scarf|scarves|shawls?)\b"),
    ("gloves",      "accessory", "accessory", r"\b(gloves?|mittens?)\b"),
    ("jewelry",     "accessory", "accessory", r"\b(necklaces?|earrings?|bracelets?|rings?|jewelry|jewellery)\b"),
    ("sunglasses",  "accessory", "accessory", r"\b(sunglasses|eyewear|glasses)\b"),
    ("socks",       "accessory", "accessory", r"\b(socks?|stockings?|tights)\b"),

    # --- ELBİSE / TULUM (tek parça) ---
    ("jumpsuit",    "dress", "dress", r"\b(jumpsuits?|playsuits?|rompers?|overalls?|dungarees?|boilersuit)\b"),
    ("dress",       "dress", "dress", r"\b(dress(?:es)?|gowns?|sundress|kaftan|caftan|frock|maxi dress|midi dress|robes?)\b"),

    # --- DIŞ GİYİM (outer) ---
    ("padded_coat", "outer", "outerwear", r"\b(puffer|padded (?:coat|jacket)|down (?:coat|jacket)|parka)\b"),
    ("coat",        "outer", "outerwear", r"\b(coats?|overcoat|trench(?:coat)?|peacoat)\b"),
    ("blazer",      "outer", "outerwear", r"\b(blazers?|suit jacket|sport coat)\b"),
    ("raincoat",    "outer", "outerwear", r"\b(raincoats?|rain jacket|windbreaker|anorak)\b"),
    ("jacket",      "outer", "outerwear", r"\b(jackets?|bomber|biker jacket|denim jacket|leather jacket|shacket)\b"),

    # --- ARA KATMAN (mid) ---
    ("cardigan",    "mid", "top", r"\b(cardigans?|kimono|ponchos?)\b"),
    ("sweater",     "mid", "top", r"\b(sweaters?|jumpers?|pullovers?|knitwear|turtleneck|knit top|knit)\b"),
    ("hoodie",      "mid", "top", r"\b(hoodies?|hooded sweatshirt)\b"),
    ("sweatshirt",  "mid", "top", r"\b(sweatshirts?)\b"),
    ("vest",        "mid", "top", r"\b(vests?|gilet|waistcoat)\b"),

    # --- ALT (bottom) ---
    ("jeans",       "bottom", "bottom", r"\b(jeans?|denim pants)\b"),
    ("shorts",      "bottom", "bottom", r"\b(shorts|bermudas?)\b"),
    ("skirt",       "bottom", "bottom", r"\b(skirts?|skorts?)\b"),
    ("leggings",    "bottom", "bottom", r"\b(leggings?|jeggings?)\b"),
    ("joggers",     "bottom", "bottom", r"\b(joggers?|sweatpants?|track ?pants?)\b"),
    ("trousers",    "bottom", "bottom", r"\b(trousers?|pants|pantalon|chinos?|slacks?|culottes?|cargo pants?|wide leg|palazzo)\b"),

    # --- ÜST / BAZ (base) ---
    ("tshirt",      "base", "top", r"\b(t-?shirts?|tees?|tee shirt)\b"),
    ("tank_top",    "base", "top", r"\b(tank tops?|tanks?|camisole|cami)\b"),
    ("bodysuit",    "base", "top", r"\b(bodysuits?)\b"),
    ("polo",        "base", "top", r"\b(polos?|polo shirt)\b"),
    ("blouse",      "mid", "top", r"\b(blouses?)\b"),
    ("shirt",       "mid", "top", r"\b(shirts?|button[\s-]?downs?|button[\s-]?ups?)\b"),
    ("tunic",       "mid", "top", r"\b(tunics?)\b"),
    ("top",         "base", "top", r"\b(crop tops?|tops?|tube top)\b"),
]

# "short sleeve(d)" / "long sleeve(d)" → kol uzunluğu; 'shorts' altı sanılmasın.
_SLEEVE_RE = re.compile(r"\b(short|long|half|three[\s-]?quarter|cap)[\s-]?sleeves?d?\b")
_COMPILED = [(sub, lr, cat, re.compile(rx, re.I)) for sub, lr, cat, rx in _RULES]


_APPAREL = {"base", "mid", "outer", "bottom", "dress"}


def classify(name: str, desc: str = "") -> dict | None:
    """Ad+açıklamadan kategori çıkar; sinyal yoksa None.

    Ürün adlarında gerçek tür çoğu zaman SON geçen giysi kelimesidir
    ("Knit Shorts"→şort, "Sweater Dress"→elbise, "Bootcut Jeans"→jean,
    "Dress with Belt"→elbise). Bu yüzden giysi (apparel) eşleşmeleri içinde
    EN SON konumdaki kazanır. Ayakkabı/aksesuar yalnızca hiç giysi kelimesi
    yoksa devreye girer (ör. "Cowboy Boots", "Chain Belt")."""
    text = f"{name or ''} {desc or ''}".lower()
    text = _SLEEVE_RE.sub(" sleeve ", text)  # "short sleeve" → "shorts" olmasın

    apparel: list[tuple[int, tuple]] = []
    footwear: list[tuple[int, tuple]] = []
    accessory: list[tuple[int, tuple]] = []
    for sub, lr, cat, rx in _COMPILED:
        m = rx.search(text)
        if not m:
            continue
        rec = (m.start(), (sub, lr, cat, m.group(0).strip()))
        if lr in _APPAREL:
            apparel.append(rec)
        elif lr == "footwear":
            footwear.append(rec)
        else:
            accessory.append(rec)

    if apparel:
        sub, lr, cat, sig = max(apparel, key=lambda r: r[0])[1]
    elif footwear:
        sub, lr, cat, sig = max(footwear, key=lambda r: r[0])[1]
    elif accessory:
        sub, lr, cat, sig = min(accessory, key=lambda r: r[0])[1]
    else:
        return None
    return {"layer_role": lr, "category": cat, "subcategory": sub, "signal": sig}


# layer_role normalizasyonu (top/base/mid karşılaştırması için)
def layer_group(layer_role: str | None) -> str:
    if layer_role in ("base", "mid", "top"):
        return "top"
    return layer_role or "?"
