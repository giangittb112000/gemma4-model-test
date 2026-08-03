#!/usr/bin/env python3
"""[Legacy] Heuristic train từ CSV + synony.

Pipeline chính: finetune/build_train_from_seeds.py (make data) —
gom finetune/data/finetune/*.json → train.json.

  python3 finetune/build_train_data.py --limit 200 --out finetune/data/train.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompt import SYSTEM_PROMPT  # noqa: E402

BRANDS = {
    "apple": "apple",
    "iphone": "apple",
    "ipad": "apple",
    "macbook": "apple",
    "mac mini": "apple",
    "macmini": "apple",
    "airpods": "apple",
    "airpod": "apple",
    "apple watch": "apple",
    "samsung": "samsung",
    "galaxy": "samsung",
    "xiaomi": "xiaomi",
    "redmi": "xiaomi",
    "poco": "xiaomi",
    "mi ": "xiaomi",
    "oppo": "oppo",
    "reno": "oppo",
    "vivo": "vivo",
    "realme": "realme",
    "nokia": "nokia",
    "sony": "sony",
    "lenovo": "lenovo",
    "asus": "asus",
    "acer": "acer",
    "dell": "dell",
    "hp": "hp",
    "msi": "msi",
    "huawei": "huawei",
    "honor": "honor",
    "nothing": "nothing",
    "nubia": "nubia",
    "oneplus": "oneplus",
    "tecno": "tecno",
    "infinix": "infinix",
    "google": "google",
    "pixel": "google",
}

CATEGORIES = [
    ("điện thoại", ["điện thoại", "dien thoai", "smartphone", "dt ", "đt "]),
    ("tablet", ["ipad", "tablet", "máy tính bảng", "pad "]),
    ("laptop", ["laptop", "macbook", "máy tính xách tay", "notebook"]),
    ("tai nghe", ["tai nghe", "airpod", "airpods", "headphone", "earbuds", "freebuds"]),
    ("đồng hồ", ["apple watch", "galaxy watch", "đồng hồ", "watch", "smartwatch", "mi band"]),
    ("loa", ["loa ", "loa bluetooth", "soundbar"]),
    ("chuột", ["chuột", "mouse"]),
    ("bàn phím", ["bàn phím", "ban phim", "keyboard"]),
    ("sạc dự phòng", ["sạc dự phòng", "pin dự phòng", "powerbank"]),
    ("cáp sạc", ["cáp", "dây sạc", "lightning", "type c", "type-c"]),
    ("ốp lưng", ["ốp lưng", "ốp ", "case "]),
    ("máy ảnh", ["máy ảnh", "camera ", "flycam", "drone"]),
    ("tivi", ["tivi", "tv ", "ti vi"]),
    ("phụ kiện", ["thẻ nhớ", "sim ", "gimbal", "micro"]),
]

IPHONE_RE = re.compile(
    r"\b(?:iphone|ip|iphon)?\s*(1[1-7]|x|xr|xs|se)(?:\s*(pro\s*max|promax|pro|plus|mini|max))?\b",
    re.I,
)
GALAXY_RE = re.compile(
    r"\b(?:galaxy|glx|samsung)?\s*(s2[0-9]|s2[0-9]\s*ultra|a\d{2}|m\d{2}|z\s*flip|z\s*fold)(?:\s*(ultra|plus|\+))?\b",
    re.I,
)
STORAGE_RE = re.compile(r"\b(128|256|512|1)(?:\s*tb|\s*t|g|gb)?\b", re.I)


def load_synonyms(path: Path) -> list[tuple[list[str], str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rules: list[tuple[list[str], str]] = []
    for line in raw:
        if "=>" not in line:
            continue
        left, right = line.split("=>", 1)
        aliases = [a.strip().lower() for a in left.split(",") if a.strip()]
        target = right.strip().lower()
        # lấy target đầu nếu có nhiều
        target = target.split(",")[0].strip()
        if aliases and target:
            rules.append((aliases, target))
    # ưu tiên alias dài trước
    rules.sort(key=lambda x: -max(len(a) for a in x[0]))
    return rules


def apply_synonym(text: str, rules: list[tuple[list[str], str]]) -> str:
    t = f" {text.lower().strip()} "
    for aliases, target in rules:
        for a in aliases:
            pat = f" {a} "
            if pat in t:
                t = t.replace(pat, f" {target} ")
                break
    return re.sub(r"\s+", " ", t).strip()


def detect_brand(q: str) -> str | None:
    ql = q.lower()
    # longer keys first
    for key in sorted(BRANDS, key=len, reverse=True):
        if key in ql:
            return BRANDS[key]
    return None


def detect_category(q: str) -> str | None:
    ql = q.lower()
    for cat, keys in CATEGORIES:
        if any(k in ql for k in keys):
            return cat
    # suy từ brand/product phổ biến
    if any(x in ql for x in ("iphone", "galaxy", "redmi", "poco", "reno", "pixel")):
        return "điện thoại"
    if "ipad" in ql or "pad " in ql:
        return "tablet"
    if "macbook" in ql or "laptop" in ql:
        return "laptop"
    if "airpod" in ql:
        return "tai nghe"
    if "watch" in ql or "band" in ql:
        return "đồng hồ"
    return None


def detect_storage(q: str) -> str | None:
    # tránh nhầm đời máy 12/13/14/15/16/17
    m = re.search(r"\b(128|256|512)(?:\s*g|\s*gb|g|gb)?\b", q.lower())
    if m:
        return f"{m.group(1)}gb"
    m = re.search(r"\b1\s*tb\b", q.lower())
    if m:
        return "1tb"
    return None


def soft_spec(q: str) -> list[str]:
    """Soft-intent → list token string (không dùng object)."""
    ql = q.lower()
    spec: list[str] = []
    if any(x in ql for x in ("pin khủng", "pin trâu", "pin lâu")):
        spec.append("battery_mah_min=5000")
    if any(x in ql for x in ("sinh viên", "giá rẻ", "giá thấp", "tầm trung thấp")):
        spec.append("price_max=20000000")
    if "dưới 10 triệu" in ql or "duoi 10 trieu" in ql:
        # ghi đè mức thấp hơn nếu vừa match sinh viên
        spec = [s for s in spec if not s.startswith("price_max=")]
        spec.append("price_max=10000000")
    if any(x in ql for x in ("gaming", "chơi game")):
        spec.append("gaming")
    if any(x in ql for x in ("camera đẹp", "chụp đẹp")):
        spec.append("camera=good")
    if "mỏng nhẹ" in ql:
        spec.append("thin_light")
    if any(x in ql for x in ("cũ", "like new", "cấn")):
        spec.append("used")
    return spec


def normalize_model_token(s: str) -> str:
    s = s.lower().strip()
    s = s.replace("promax", "pro max").replace("pro max", "pro max")
    s = re.sub(r"\s+", " ", s)
    return s


def parse_iphone(q: str) -> tuple[str | None, str | None, str | None]:
    """product, brand, model"""
    ql = q.lower()
    if "iphone" not in ql and not re.search(r"\bip\d", ql) and "ip " not in ql:
        # "17 pro max", "16 pro" standalone phổ biến trong search VN
        m = re.search(
            r"\b(1[1-7])\s*(pro\s*max|promax|pro|plus|mini)?\b", ql
        )
        if m and not any(
            x in ql for x in ("xiaomi", "redmi", "galaxy", "samsung", "oppo", "reno")
        ):
            ver = m.group(1)
            variant = normalize_model_token(m.group(2) or "")
            model = f"{ver} {variant}".strip()
            return "iphone", "apple", model or ver
        return None, None, None

    m = IPHONE_RE.search(ql)
    if not m:
        if "iphone" in ql:
            return "iphone", "apple", None
        return None, None, None
    ver = m.group(1).lower()
    variant = normalize_model_token(m.group(2) or "")
    model = f"{ver} {variant}".strip()
    return "iphone", "apple", model or None


def parse_samsung(q: str) -> tuple[str | None, str | None, str | None]:
    ql = q.lower()
    if not any(x in ql for x in ("samsung", "galaxy", "glx", "s25", "s24", "s23", "a5", "a1", "m3", "m5")):
        return None, None, None
    m = GALAXY_RE.search(ql)
    product = "galaxy"
    brand = "samsung"
    if m:
        model = normalize_model_token(m.group(0))
        model = (
            model.replace("samsung", "")
            .replace("galaxy", "")
            .replace("glx", "")
            .strip()
        )
        return product, brand, model or None
    if "samsung" in ql or "galaxy" in ql:
        return product, brand, None
    return None, None, None


def label_query(raw_query: str, rules: list[tuple[list[str], str]]) -> dict:
    q0 = raw_query.strip()
    q = apply_synonym(q0, rules)

    spec = soft_spec(q)
    storage = detect_storage(q)
    if storage:
        spec.insert(0, storage)

    category = detect_category(q)
    brand = detect_brand(q)
    product: str | None = None
    model: str | None = None

    p, b, m = parse_iphone(q)
    if p:
        product, brand, model = p, b, m
        category = category or "điện thoại"

    if product is None:
        p, b, m = parse_samsung(q)
        if p:
            product, brand, model = p, b or brand, m
            category = category or "điện thoại"

    # xiaomi / redmi / poco
    ql = q.lower()
    if product is None and any(x in ql for x in ("xiaomi", "redmi", "poco", "mi pad")):
        brand = brand or "xiaomi"
        if "pad" in ql:
            category = category or "tablet"
            product = "pad"
        elif "redmi" in ql:
            category = category or "điện thoại"
            product = "redmi"
        elif "poco" in ql:
            category = category or "điện thoại"
            product = "poco"
        else:
            # chỉ "xiaomi" → brand, không đặt product=xiaomi
            category = category or "điện thoại"
            product = None
        mm = re.search(
            r"(?:redmi|poco|xiaomi)\s+([a-z0-9][a-z0-9\s\-]{0,20})", ql
        )
        if mm:
            model = normalize_model_token(mm.group(1))
            if product is None and "redmi" not in ql and "poco" not in ql:
                product = None  # model dòng máy xiaomi nếu có

    # oppo reno
    if product is None and ("oppo" in ql or "reno" in ql):
        brand = brand or "oppo"
        category = category or "điện thoại"
        product = "reno" if "reno" in ql else "oppo"
        mm = re.search(r"reno\s*([0-9a-z]+)", ql)
        if mm:
            model = f"reno{mm.group(1)}"

    # ipad / macbook / airpods / watch
    if "ipad" in ql:
        category = category or "tablet"
        product = product or "ipad"
        brand = brand or "apple"
        mm = re.search(
            r"ipad\s*(air|pro|mini)?(?:\s*([0-9][0-9a-z\.]*|m\d))?", ql
        )
        if mm:
            parts = [x for x in (mm.group(1), mm.group(2)) if x]
            model = " ".join(parts) or None
    if "macbook" in ql:
        category = category or "laptop"
        product = product or "macbook"
        brand = brand or "apple"
    if "airpod" in ql:
        category = category or "tai nghe"
        product = product or "airpods"
        brand = brand or "apple"
    if "apple watch" in ql or re.search(r"\baw\b", ql):
        category = category or "đồng hồ"
        product = product or "apple watch"
        brand = brand or "apple"

    # chỉ hãng (samsung, xiaomi, …) → product = tên hãng (schema không còn brand)
    if product is None and brand:
        if brand in ("samsung", "xiaomi", "oppo", "vivo", "realme", "apple", "dell", "lenovo"):
            category = category or (
                "điện thoại"
                if brand in ("samsung", "xiaomi", "oppo", "vivo", "realme", "apple")
                else None
            )
            product = brand

    # category-only keywords
    if category is None:
        for cat, keys in CATEGORIES:
            if any(k.strip() == ql or ql.startswith(k.strip()) for k in keys):
                category = cat
                break
        if ql in ("tai nghe", "laptop", "chuột", "bàn phím", "loa", "máy ảnh"):
            category = {
                "tai nghe": "tai nghe",
                "laptop": "laptop",
                "chuột": "chuột",
                "bàn phím": "bàn phím",
                "loa": "loa",
                "máy ảnh": "máy ảnh",
            }[ql]

    # gộp đời máy vào product (schema: category / product / spec)
    if product and model:
        product = f"{product} {model}".strip()
    elif model and not product:
        product = model

    return {
        "category": category,
        "product": product,
        "spec": spec,
    }


def to_messages(query: str, output: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f'query: "{query}"'},
            {
                "role": "model",
                "content": json.dumps(output, ensure_ascii=False, separators=(",", ":")),
            },
        ]
    }


def synonym_variants(
    query: str, rules: list[tuple[list[str], str]], limit: int = 2
) -> list[str]:
    """Alias từ synony: rule có target≈query, hoặc query nằm trong aliases."""
    ql = query.lower().strip()
    out: list[str] = []
    for aliases, target in rules:
        # Chỉ rule đúng khái niệm: target trùng query, hoặc query là 1 alias trong nhóm.
        related = target == ql or ql in aliases
        if not related:
            continue
        for a in aliases:
            a = a.strip()
            if not a or a == ql or len(a) < 2 or len(a) > 36:
                continue
            # bỏ alias quá chung / câu dài kiểu mô tả
            if a.count(" ") >= 5:
                continue
            out.append(a)
            if len(out) >= limit:
                return out
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=ROOT / "data" / "search_report.csv")
    ap.add_argument("--synony", type=Path, default=ROOT / "data" / "synony.json")
    ap.add_argument("--out", type=Path, default=ROOT / "finetune" / "data" / "train.json")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--with-synonym-variants", action="store_true", default=True)
    ap.add_argument("--max-total", type=int, default=350, help="Giới hạn tổng mẫu (keyword+variant)")
    args = ap.parse_args()

    rules = load_synonyms(args.synony)
    keywords: list[str] = []
    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term = (row.get("search_term") or "").strip()
            if term:
                keywords.append(term)
            if len(keywords) >= args.limit:
                break

    samples: list[dict] = []
    seen_q: set[str] = set()

    for kw in keywords:
        key = kw.lower()
        if key in seen_q:
            continue
        seen_q.add(key)
        out = label_query(kw, rules)
        samples.append(to_messages(kw, out))

        if args.with_synonym_variants and len(samples) < args.max_total:
            for alias in synonym_variants(kw, rules):
                ak = alias.lower()
                if ak in seen_q:
                    continue
                seen_q.add(ak)
                labeled = label_query(alias, rules)
                samples.append(to_messages(alias, labeled))
                if len(samples) >= args.max_total:
                    break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    filled = 0
    for s in samples:
        o = json.loads(s["messages"][-1]["content"])
        if o.get("category") or o.get("product"):
            filled += 1
    print(f"[ok] keywords={len(keywords)} samples={len(samples)} -> {args.out}")
    print(f"     có category/product: {filled}/{len(samples)}")


if __name__ == "__main__":
    main()
