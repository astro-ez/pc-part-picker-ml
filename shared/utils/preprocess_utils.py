import unicodedata
from typing import List, Dict, Tuple
import re

CATEGORIES = ["CASE","CASE_FAN","CPU","CPU_COOLER","GPU","MOTHERBOARD","PSU","RAM","STORAGE"]

# 1) French->English term mapping for PC parts (extend as you see them)
FR_EN_MAP = {
    # core nouns
    "boitier": "case",
    "boîtier": "case",
    "tour": "case",  # "tour pc" often means case
    "ventilateur": "fan",
    "ventilateurs": "fans",
    "ventilo": "fan",
    "processeur": "cpu",
    "carte mere": "motherboard",
    "carte mère": "motherboard",
    "alimentation": "psu",
    "memoire": "ram",
    "mémoire": "ram",
    "barrette": "ram",
    "disque dur": "hdd",
    "disque": "disk",
    "ssd": "ssd",
    "carte graphique": "gpu",
    "refroidisseur": "cooler",
    "refroidissement": "cooling",
    "watercooling": "aio",
    "pate thermique": "thermal paste",
    "pâte thermique": "thermal paste",
    "boitier ventilateur": "case fan",
    "ventilateur boitier": "case fan",
    # marketing/noise
    "avec": "with",
    "pack": "bundle",
    "lot": "bundle",
    "kit": "bundle",
    "promo": "",
    "offert": "free",
    "rgb": "rgb",
    "argb": "argb",
    "edition": "edition",
    "édition": "edition",
    "jeux": "game",
}

# 2) Lexicons for category hints (include common EN + mapped FR tokens)
LEX = {
    "CASE": [
        "case","mid tower","full tower","mini tower","atx tower","matx tower","itx tower",
        "h510","p400a","meshify","lan li","lian li","nzxt","fractal","h7","o11","o11d"
    ],
    "CASE_FAN": [
        "case fan","fan","fans","120mm","140mm","200mm","p12","p14","af120","q120","ml120","tlc fan"
    ],
    "CPU": [
        "cpu","ryzen","intel","i3","i5","i7","i9","xeon","5800x","5600x","7800x3d","12400","13600k"
    ],
    "CPU_COOLER": [
        "cooler","air cooler","aio","watercooling","liquid","hyper 212","dual tower","nh-d15","kraken","castle","iceberg"
    ],
    "GPU": [
        "gpu","rtx","gtx","rx","radeon","geforce","4060","4070","4080","4090","7600","7800","7900","ti","super"
    ],
    "MOTHERBOARD": [
        "motherboard","mobo","b550","x570","x670","z490","z690","z790","b660","b760","a620","b650","h610","h670",
        "atx","matx","micro-atx","micro atx","mini-itx","itx"
    ],
    "PSU": [
        "psu","power supply","alimentation","rm750","rm850","cx650","700w","750w","850w","1000w","80+","80 plus","gold","bronze","platinum"
    ],
    "RAM": [
        "ram","ddr4","ddr5","sodimm","udimm","trident","vengeance","ripjaws","32gb","16gb","8gb","6000mhz","3200mhz"
    ],
    "STORAGE": [
        "ssd","hdd","nvme","m.2","sata","sn850","970 evo","980 pro","mx500","blue","ironwolf","barracuda","exos","p3","p5"
    ],
}

# Size/form-factor tokens to help distinguish CASE vs MOBO
FORM_FACTORS = ["atx","matx","micro-atx","micro atx","mini-itx","itx","e-atx","eatx"]

# Bundle indicators
BUNDLE_TOKENS = ["+", " bundle", " kit", " with ", " x ", " lot", " pack", "combo", "pack ", "avec ", "et "]

# Noise tokens to strip (kept lowercase)
NOISE = [
    "new","neuf","original","genuine","edition","special","rgb","argb","white","black","noir","blanc",
    "gaming","gamer","pc","desktop","pour","pour pc","version","2024","2025","promo","offert","deal"
]

# Priority of main item if multiple components appear (tune to your needs)
MAIN_PRIORITY = ["GPU","CPU","MOTHERBOARD","RAM","STORAGE","PSU","CPU_COOLER","CASE_FAN","CASE"]

def strip_accents(text: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def normalize_title(raw: str) -> str:
    t = raw.lower()
    t = strip_accents(t)
    t = t.replace("‐", "-").replace("–", "-").replace("—", "-")
    # cheap punctuation spacing
    t = re.sub(r"([+/,()])", r" \1 ", t)
    t = normalize_spaces(t)
    # translate FR->EN terms (word-level and phrase-level)
    # do longer keys first to avoid partial collisions
    for k in sorted(FR_EN_MAP.keys(), key=len, reverse=True):
        t = re.sub(rf"\b{re.escape(k)}\b", FR_EN_MAP[k], t)
    # remove repeated spaces and obvious noise words
    words = [w for w in t.split() if w not in NOISE]
    t = " ".join(words)
    return t

def detect_bundle(text: str) -> bool:
    t = " " + text + " "
    return any(tok in t for tok in BUNDLE_TOKENS)

def collect_component_hints(text: str) -> Dict[str, int]:
    """Count hits per category based on lexicons."""
    counts = {c:0 for c in CATEGORIES}
    t = " " + text + " "
    for cat, kws in LEX.items():
        for kw in kws:
            # word-ish match; allow model names (numbers) to hit
            if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", t):
                counts[cat] += 1
    # heuristic: motherboard FF present → boost MOTHERBOARD; if 'tower' present → boost CASE
    if any(ff in t for ff in FORM_FACTORS):
        counts["MOTHERBOARD"] += 1
    if "tower" in t:
        counts["CASE"] += 1
    # avoid counting generic 'fan' as CASE_FAN if it's an AIO context
    if "aio" in t or "liquid" in t or "watercool" in t:
        counts["CPU_COOLER"] += 1
    return counts

def guess_main_label(counts: Dict[str,int]) -> str:
    # pick by count; tie-break by priority
    best = None
    best_count = -1
    for cat in CATEGORIES:
        c = counts.get(cat, 0)
        if c > best_count or (c == best_count and
                              (best is None or MAIN_PRIORITY.index(cat) < MAIN_PRIORITY.index(best))):
            best = cat
            best_count = c
    return best or "STORAGE"  # safe default

def parse_components(text: str) -> List[str]:
    """
    Extract short component hints from a bundle title.
    Very light heuristic: split by '+' or 'with' or 'bundle'.
    """
    t = text
    t = t.replace("/", " / ").replace("+", " + ")
    parts = re.split(r"\b(?:\+|with|bundle|kit|pack|combo|lot|et|,)\b", t)
    parts = [normalize_spaces(p) for p in parts if len(p.strip()) >= 3]
    # keep short phrases
    return [p[:60] for p in parts][:6]
    
def preprocess_title(raw_title: str) -> Dict:
    t_norm = normalize_title(raw_title)
    is_bundle = detect_bundle(t_norm)
    counts = collect_component_hints(t_norm)
    main_label = guess_main_label(counts)
    components = parse_components(t_norm) if is_bundle else []
    return {
        "normalized_title": t_norm,
        "bundle": is_bundle,
        "component_hints": components,
        "lexicon_counts": counts,
        "main_label_guess": main_label,
    }
