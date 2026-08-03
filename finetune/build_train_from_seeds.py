#!/usr/bin/env python3
"""Gom finetune/data/finetune/*.json → finetune/data/train.json (TRL messages).

Mỗi seed file: list object
  {"query","category","product","spec":[]}

  python3 finetune/build_train_from_seeds.py
  python3 finetune/build_train_from_seeds.py --seeds finetune/data/finetune --out finetune/data/train.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from prompt import SYSTEM_PROMPT  # noqa: E402


def normalize_spec(spec) -> list[str]:
    if not spec:
        return []
    if isinstance(spec, dict):
        # legacy object → list token
        out: list[str] = []
        for k, v in spec.items():
            if v is None:
                continue
            if k in ("storage", "ram") or isinstance(v, str) and v.endswith("gb"):
                out.append(str(v).lower().replace(" ", ""))
            else:
                out.append(f"{k}={v}")
        return out
    if not isinstance(spec, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for s in spec:
        t = str(s).strip().lower()
        t = t.replace(" gb", "gb").replace(" hz", "hz")
        t = " ".join(t.split())
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def load_seeds(seed_dir: Path) -> list[dict]:
    if not seed_dir.is_dir():
        raise SystemExit(f"Không thấy seed dir: {seed_dir}")

    rows: list[dict] = []
    for path in sorted(seed_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"[warn] skip {path.name}: {exc}", file=sys.stderr)
            continue
        if not isinstance(data, list):
            print(f"[warn] skip {path.name}: không phải list", file=sys.stderr)
            continue
        for item in data:
            if not isinstance(item, dict):
                continue
            q = (item.get("query") or "").strip()
            if not q:
                continue
            rows.append(
                {
                    "query": q,
                    "category": item.get("category"),
                    "product": item.get("product"),
                    "spec": normalize_spec(item.get("spec")),
                    "source": path.stem,
                }
            )
    return rows


def dedupe_by_query(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for r in rows:
        key = r["query"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def to_messages(query: str, category, product, spec: list[str]) -> dict:
    output = {
        "category": category,
        "product": product,
        "spec": list(spec),
    }
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds",
        type=Path,
        default=ROOT / "data" / "finetune",
        help="Thư mục *.json seed (query/category/product/spec)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "train.json")
    ap.add_argument(
        "--no-dedupe",
        action="store_true",
        help="Giữ query trùng giữa các file seed",
    )
    args = ap.parse_args()

    rows = load_seeds(args.seeds)
    before = len(rows)
    if not args.no_dedupe:
        rows = dedupe_by_query(rows)

    samples = [
        to_messages(r["query"], r["category"], r["product"], r["spec"]) for r in rows
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    filled = sum(1 for r in rows if r["category"] or r["product"] or r["spec"])
    with_spec = sum(1 for r in rows if r["spec"])
    print(f"[ok] seed_files={args.seeds} raw={before} unique={len(rows)} -> {args.out}")
    print(f"     có label: {filled}/{len(rows)} | có spec: {with_spec}")


if __name__ == "__main__":
    main()
