#!/usr/bin/env python3
"""Üretim envanteri ve kombin üretimi için parça uygunluk kuralları."""

from __future__ import annotations

import re

NON_STREET_WEAR_KEYWORDS = (
    "pajama", "pyjama", "nightgown", "nightdress", "night dress", "sleepwear", "nightwear", "shapewear",
    "loungewear", "lounge set", "lounge nightgown", "blanket hoodie",
    "corset", "waist trainer", "lingerie", "bralette", "underwire",
    "panty", "panties", "thong", "briefs", "underwear", "undergarment",
    "shaping panty", "shaping short", "control brief", "intimates",
    "sports bra", "bra top", "nursing bra",
    "keychain", "coin purse", "phone case", "hair clip", "hairpin",
    "sunglass", "eyewear", "glasses", "wallet chain",
)

BEACH_SWIM_KEYWORDS = (
    "bikini", "swim", "pareo", "paréo", "cover-up", "cover up", "beach cover",
    "swimsuit", "swim dress", "swim short", "halter neck bikini", "kimono beach",
    "beach kimono", "one-piece swim", "swim set", "triki", "beach wear", "tankini",
)

NON_WEARABLE_ACCESSORY_KEYWORDS = (
    "handbag", "hand bag", "shoulder bag", " tote", "clutch", " crossbody",
    " satchel", "backpack", " wig", "purse", "wallet", "belt buckle",
)

NON_GARMENT_KEYWORDS = (
    "deodorant", "body mist", "body spray", "cologne", "fragrance gift",
    "perfume", "parfum", "parfüm", "eau de toilette", "eau de parfum",
    "nail polish", "nail lacquer", "lipstick", "lip gloss", "lip liner",
    "makeup", "foundation", "concealer", "compact", "eyeshadow", "kajal",
    "eyeliner", "highlighter", "blush", "moisturiser", "moisturizer",
    "shampoo", "conditioner", "beauty accessory", "body shimmer",
    "cosmetic bag", "free gift", "laptop bag", "mobile pouch", "duffel bag",
    "messenger bag", "trolley bag", "suitcase", "stationery",
    "watch", "wristwatch", "earring", "necklace", "pendant", "bracelet",
    "bangle", "cufflink", "jewellery set", "jewelry set", "accessory gift set",
)

CHARACTER_KEYWORDS = (
    "dora", "barbie", "disney", "frozen", "spiderman", "spider-man", "spider man",
    "minnie", "mickey", "hello kitty", "peppa", "paw patrol", "pokemon", "pokémon",
    "looney tunes", "winnie the pooh", "elsa", "avengers", "batman", "superman",
    "captain america", "ironman", "iron man", "cartoon", "ninja turtles",
    "my little pony", "hot wheels", "cocomelon", "bluey",
)

_FP_TYPE_RE = re.compile(r":\s*([^,]+?),", re.I)
_FP_TYPE_NORM_RE = re.compile(
    r"\s+(casual|formal|sports|ethnic|smart|party|wedding)\s+.*$",
    re.I,
)

NON_WEARABLE_FP_ARTICLE_TYPES = frozenset({
    "watches", "handbags", "sunglasses", "wallets", "backpacks",
    "perfume and body mist", "deodorant", "earrings", "clutches",
    "nail polish", "lipstick", "pendant", "necklace and chains",
    "trunk", "ring", "lip gloss", "cufflinks", "accessory gift set",
    "kajal and eyeliner", "free gifts", "duffel bag", "bangle",
    "laptop bag", "foundation and primer", "jewellery set",
    "fragrance gift set", "face moisturisers", "mobile pouch",
    "lip liner", "messenger bag", "compact", "eyeshadow",
    "highlighter and blush", "beauty accessory", "nail lacquer",
    "makeup", "nightdress", "bra", "briefs", "boxers", "panties",
    "ties and cufflinks", "trolley bag", "suitcase", "travel accessory",
    "stationery", "key chain", "keychain", "gift set", "body mist",
    "perfumes", "perfume", "deos", "deo set",
    "innerwear vests", "camisoles", "socks", "stockings", "booties",
    "flip flops", "lounge pants", "lounge shorts", "lounge tshirts",
    "night suits", "baby dolls", "shrug",
})


def text_blob(garment: dict) -> str:
    return f"{garment.get('name', '')} {garment.get('description', '')}".lower()


def fp_article_type_core(garment: dict) -> str | None:
    """MIT 44K açıklamasından ürün türünü çıkar: '…: Deodorant Casual Women, …'."""
    m = _FP_TYPE_RE.search(garment.get("description", ""))
    if not m:
        return None
    core = _FP_TYPE_NORM_RE.sub("", m.group(1).strip().lower()).strip()
    return core or None


NON_GARMENT_MASTER_CATEGORIES = frozenset({
    "Personal Care", "Free Items", "Home", "Sporting Goods",
})


def _fp_master_category(garment: dict) -> str | None:
    meta = garment.get("fp_meta") or garment.get("source_metadata") or {}
    return meta.get("masterCategory")


ETHNIC_KEYWORDS = (
    "saree", "sari ", " sari", "lehenga", "lehanga", "kurta", "kurti", "salwar",
    "churidar", "dupatta", "sherwani", "dhoti", "lungi", "anarkali", "ghagra",
    "choli", "patiala", "kaftan", "kalidar", "angrakha", "jodhpuri", "mundu",
    "ethnic", "jaipur print", "bhagalpur", "nehru jacket",
)


def is_ethnic_wear(garment: dict) -> bool:
    meta = garment.get("fp_meta") or garment.get("source_metadata") or {}
    if (meta.get("usage") or "").strip().lower() == "ethnic":
        return True
    blob = text_blob(garment)
    return any(kw in blob for kw in ETHNIC_KEYWORDS)


def non_garment_reason(garment: dict) -> str | None:
    master = _fp_master_category(garment)
    if master in NON_GARMENT_MASTER_CATEGORIES:
        return f"non_garment_master:{master}"
    if is_ethnic_wear(garment):
        return "ethnic_wear"
    name = (garment.get("name") or "").strip().lower()
    article = fp_article_type_core(garment)
    # Kemer (kıyafet değil aksesuar): adında "belt" geçer ama gerçek bir giysi
    # kelimesi yoktur. "belted ..." (sıfat) \bbelt\b ile eşleşmez, korunur.
    if article in ("belts", "belt"):
        return "accessory_belt"
    if re.search(r"\bbelts?\b", name) and not re.search(
        r"\b(dress|pant|trouser|jean|jumpsuit|romper|skirt|short|gown|tunic|"
        r"top|shirt|blouse|jacket|coat|sweater|cardigan|hoodie|blazer|vest|"
        r"legging|playsuit|overall|dungaree)\b",
        name,
    ):
        return "accessory_belt"
    # Ayakkabı bakım/aksesuar (fırça, bağcık, taban, çekecek)
    blob_lc = text_blob(garment)
    if (
        article in ("shoe accessories", "shoe laces", "shoe care")
        or re.search(
            r"shoe (accessor|lace|brush|care|horn|tree|polish|whitener)|"
            r"shoelace|\binsole|shoe bag",
            blob_lc,
        )
    ):
        return "non_garment:shoe_accessory"
    blob = text_blob(garment)
    if re.search(r"\b(hair colou?r|hair dye|hair cream|skin cream|body lotion|"
                 r"face wash|cleanser|moistur|scrub|serum|sunscreen|shampoo|"
                 r"conditioner|talc|peel off|face pack)\b", blob):
        return "non_garment:personal_care"
    if re.search(r"\bdeo\b", blob):
        return "non_garment:deo"
    for kw in NON_GARMENT_KEYWORDS:
        if kw in blob:
            return f"non_garment:{kw}"
    article = fp_article_type_core(garment)
    if article and article in NON_WEARABLE_FP_ARTICLE_TYPES:
        return f"non_garment_type:{article}"
    if article and any(
        article.startswith(p)
        for p in ("perfume", "deodorant", "lipstick", "nail ", "makeup", "beauty")
    ):
        return f"non_garment_type:{article}"
    return None


def exclusion_reason(garment: dict) -> str | None:
    """Sokak kombini / üretim gardırobu dışı bırakma nedeni."""
    if garment.get("active") is False:
        return "inactive"
    blob = text_blob(garment)
    for kw in NON_STREET_WEAR_KEYWORDS:
        if kw in blob:
            return f"non_street:{kw}"
    for kw in BEACH_SWIM_KEYWORDS:
        if kw in blob:
            return f"beach_swim:{kw}"
    for kw in NON_WEARABLE_ACCESSORY_KEYWORDS:
        if kw in blob:
            return f"accessory_junk:{kw}"
    junk = non_garment_reason(garment)
    if junk:
        return junk
    for kw in CHARACTER_KEYWORDS:
        if kw in blob:
            return f"character_print:{kw}"
    from garment_gender import is_kids_item
    if is_kids_item(garment):
        return "kids"
    if garment.get("layer_role") == "accessory" and garment.get("subcategory") == "scarf":
        if any(k in blob for k in ("sunglass", "keychain", "wallet", "bag", "glasses")):
            return "mislabeled_accessory"
    if garment.get("layer_role") == "dress" and any(
        k in blob for k in ("sandal", "heel", "shoe", "boot", "sneaker")
    ):
        return "footwear_as_dress"
    rel = garment.get("image_path", "")
    if not rel:
        return "no_image"
    return None


def is_catalog_eligible(garment: dict, *, require_image: bool = True) -> bool:
    """Üretim gardırobunda tutulabilir mi?"""
    if exclusion_reason(garment) is not None:
        return False
    if require_image:
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        if not (root / garment.get("image_path", "")).exists():
            return False
    return True
