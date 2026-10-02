"""End-to-end check of the browser demo (Edge via Playwright) against PyTorch.

1. Every sample texture: browser output vs PyTorch output on the same input (single tile).
2. A 300x300 upload (forces 3x3 tiling): browser tiled output vs PyTorch full-image output,
   which would expose tile seams.
Writes eval/browser_check.json.

  python -m http.server 8601 --directory docs     # other shell
  python deploy/test_web.py http://127.0.0.1:8601/ --model rrdb_s_gan
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.models import build_generator  # noqa: E402

GRAB = "() => { const c = window.pixelforge.sr; return c.toDataURL('image/png'); }"


def torch_sr(net, img):
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        y = net(x).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    return (y * 255).round().astype(np.uint8)


def grab(pg):
    url = pg.evaluate(GRAB)
    return np.array(Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    ck = torch.load(ROOT / "models" / f"{args.model}.pt", map_location="cpu", weights_only=False)
    net = build_generator(**ck["config"]).eval()
    net.load_state_dict(ck["ema"])
    meta = json.loads((ROOT / "docs" / "meta.json").read_text())
    res = {"samples": [], "tiled": None}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1280, "height": 1000})
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(args.url, wait_until="networkidle")
        for i, s in enumerate(meta["samples"]):
            pg.locator("#thumbs .thumb").nth(i).click()
            pg.wait_for_function(f"window.pixelforge.results['{s['lr']}'] !== undefined", timeout=180000)
            got = grab(pg)
            ref = torch_sr(net, np.array(Image.open(ROOT / "docs" / s["lr"]).convert("RGB")))
            diff = np.abs(got.astype(int) - ref.astype(int))
            ms = pg.evaluate(f"window.pixelforge.results['{s['lr']}'].ms")
            res["samples"].append({"name": s["name"], "mean_abs_diff": float(diff.mean()), "max_abs_diff": int(diff.max()), "ms": ms})
            print(f"{s['name']:28s} mean|diff|={diff.mean():.3f} max={diff.max()} {ms:.0f} ms")
        # tiled upload
        rng = np.random.default_rng(0)
        src = np.array(Image.open(ROOT / "docs" / meta["samples"][0]["hr"]).convert("RGB").resize((300, 300), Image.BICUBIC))
        src = np.clip(src.astype(int) + rng.integers(-3, 4, src.shape), 0, 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            Image.fromarray(src).save(f.name)
        pg.set_input_files("#file", f.name)
        pg.wait_for_function("document.getElementById('status').textContent.includes('in your browser') && window.pixelforge.sr.width === 1200", timeout=300000)
        got = grab(pg)
        ref = torch_sr(net, src)
        diff = np.abs(got.astype(int) - ref.astype(int))
        res["tiled"] = {"input": "300x300", "mean_abs_diff": float(diff.mean()), "p99_abs_diff": float(np.percentile(diff, 99)),
                        "max_abs_diff": int(diff.max()), "status": pg.inner_text("#status")}
        print(f"tiled 300x300 -> 1200x1200: mean|diff|={diff.mean():.3f} p99={np.percentile(diff, 99):.1f} max={diff.max()}")
        res["console_errors"] = errs
        b.close()
    (ROOT / "eval" / "browser_check.json").write_text(json.dumps(res, indent=2))
    print("console errors:", errs)


if __name__ == "__main__":
    main()
