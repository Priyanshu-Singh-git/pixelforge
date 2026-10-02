"""Download Poly Haven CC0 texture albedo maps (1k JPG) and make a split by texture.

All Poly Haven assets are CC0 (public domain): https://polyhaven.com/license
Writes data/polyhaven/<slug>.jpg and data/manifest.json (slug, name, categories, split).
"""
from __future__ import annotations

import json
import random
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "polyhaven"
API = "https://api.polyhaven.com"
UA = {"User-Agent": "pixelforge-research/1.0"}


def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def fetch(slug):
    dst = OUT / f"{slug}.jpg"
    if dst.exists() and dst.stat().st_size > 0:
        return slug, True
    try:
        files = get_json(f"{API}/files/{slug}")
        url = files["Diffuse"]["1k"]["jpg"]["url"]
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
            dst.write_bytes(r.read())
        return slug, True
    except Exception as e:  # some assets have no diffuse map
        return slug, f"{type(e).__name__}: {e}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assets = get_json(f"{API}/assets?t=textures")
    with ThreadPoolExecutor(16) as ex:
        res = dict(ex.map(fetch, sorted(assets)))
    ok = sorted(s for s, r in res.items() if r is True)
    rng = random.Random(0)
    order = ok[:]
    rng.shuffle(order)
    split = {s: "test" for s in order[:60]} | {s: "val" for s in order[60:120]} | {s: "train" for s in order[120:]}
    manifest = [{"slug": s, "name": assets[s]["name"], "categories": assets[s].get("categories", []),
                 "split": split[s], "license": "CC0 (polyhaven.com)"} for s in ok]
    (ROOT / "data" / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"downloaded {len(ok)}/{len(assets)}; failed: {[s for s, r in res.items() if r is not True][:5]}")
    print({k: sum(m['split'] == k for m in manifest) for k in ('train', 'val', 'test')})


if __name__ == "__main__":
    main()
