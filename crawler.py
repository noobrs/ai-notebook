#!/usr/bin/env python3
"""
Shopee Review Crawler (Malay–English e‑commerce reviews)
-------------------------------------------------------
• Input: text file with either full Shopee product URLs **or** plain item IDs, one per line.
• Output: CSV of reviews with star‑rating, comment, and a rough language guess.
• Uses Shopee public item/get_ratings JSON endpoint (undocumented, so use responsibly).
• Designed for coursework / research; **do not** hammer Shopee — default rate‑limit = 1 req/sec.

Changelog
~~~~~~~~~
2025‑07‑20 • Improved URL parsing: now accepts query strings and captures the second
             numeric group after the "-i." marker (shopid.itemid). Fallback to
             `itemid=` query parameter or final numeric segment.
"""
from __future__ import annotations

import re
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any

import requests
from langdetect import detect_langs

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# URL / ID helpers
# ---------------------------------------------------------------------------

_ITEMID_PATTERNS = [
    # Pattern 1: shopee.com.my/<slug>-i.<shopid>.<itemid>[?...]
    re.compile(r"-i\\.(?P<shopid>\\d+)\\.(?P<itemid>\\d+)", re.IGNORECASE),
    # Pattern 2: ...?itemid=<itemid>&...
    re.compile(r"[?&]itemid=(?P<itemid>\\d+)", re.IGNORECASE),
    # Pattern 3: trailing numeric segment before optional query string
    re.compile(r"\\.(?P<itemid>\\d+)(?:$|\\?)"),
]

def extract_itemid(s: str) -> int:
    """Return numeric **itemid** from a Shopee product URL or raise ValueError."""
    if s.isdigit():
        return int(s)

    for pat in _ITEMID_PATTERNS:
        m = pat.search(s)
        if m and "itemid" in m.groupdict():
            return int(m.group("itemid"))
    raise ValueError(f"Could not parse itemid from: {s}")

# ---------------------------------------------------------------------------
# Network logic
# ---------------------------------------------------------------------------

def fetch_ratings(itemid: int, max_pages: int = 50, delay: float = 1.0) -> List[Dict[str, Any]]:
    """Fetch up to (max_pages * 20) review entries for one itemid."""
    limit = 20  # Shopee endpoint fixed page size
    reviews: List[Dict[str, Any]] = []
    for page in range(max_pages):
        offset = page * limit
        api = (
            "https://shopee.com.my/api/v2/item/get_ratings?"
            f"filter=0&flag=1&itemid={itemid}&limit={limit}&offset={offset}&type=0"
        )
        try:
            r = requests.get(api, headers=HEADERS, timeout=10)
        except requests.RequestException as exc:
            print(f"⚠️ Network error on {api}: {exc}")
            break
        if r.status_code != 200:
            print(f"⚠️ HTTP {r.status_code} on {api}")
            break
        try:
            data = r.json()
        except json.JSONDecodeError:
            print("⚠️ Non‑JSON response — stopping")
            break
        items = data.get("data", {}).get("ratings", [])
        if not items:
            break  # no more pages

        for it in items:
            comment = (it.get("comment") or "").replace("\n", " ").strip()
            if not comment:
                continue  # skip blank comments
            try:
                lang_probs = detect_langs(comment)
                lang_guess = ",".join(f"{lp.lang}:{lp.prob:.2f}" for lp in lang_probs)
            except Exception:
                lang_guess = "unknown"
            reviews.append(
                {
                    "itemid": itemid,
                    "rating": it.get("rating_star"),
                    "comment": comment,
                    "language_guess": lang_guess,
                }
            )
        time.sleep(delay)
    return reviews

# ---------------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Shopee review crawler ⇒ CSV")
    parser.add_argument(
        "input", metavar="INPUT.txt", help="Text file of product URLs *or* item IDs, one per line"
    )
    parser.add_argument("--out", default="reviews.csv", help="CSV output filename")
    parser.add_argument("--pages", type=int, default=50, help="Max pages per item (×20 reviews)")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="Seconds to sleep between successive requests"
    )
    args = parser.parse_args()

    targets = [ln.strip() for ln in Path(args.input).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not targets:
        print("⚠️ No targets found in input file.")
        return

    all_rows: List[Dict[str, Any]] = []
    for t in targets:
        try:
            itemid = extract_itemid(t)
        except ValueError as e:
            print("⚠️", e)
            continue
        print(f"▶ Crawling itemid {itemid}")
        rows = fetch_ratings(itemid, max_pages=args.pages, delay=args.delay)
        print(f"   → {len(rows)} rows")
        all_rows.extend(rows)

    print(f"Total collected: {len(all_rows)} rows")
    if not all_rows:
        print("Nothing to write – exiting.")
        return

    keys = ["itemid", "rating", "comment", "language_guess"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"✅ Saved to {args.out}")

if __name__ == "__main__":
    main()
