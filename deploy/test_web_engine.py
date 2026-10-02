"""Browser engine features vs the Python engine (Edge via Playwright).

  python deploy/test_web_engine.py http://127.0.0.1:8601/
1. seamless mode: browser output vs `pixelforge upscale --tileable`, and wrap-edge seam of both
2. transparency: an RGBA upload comes back RGBA with the same alpha footprint
3. batch: 3 uploads -> one ZIP with 3 x4 PNGs
Appends results to eval/browser_check.json under "engine".
"""
from __future__ import annotations

import base64
import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))
from pixelforge.data import build_eval_set  # noqa: E402
from pixelforge.toolkit import DEFAULT_MODEL, load_model, upscale_rgb  # noqa: E402
from toolkit_checks import seam  # noqa: E402

GRAB = "() => window.pixelforge.sr.toDataURL('image/png')"


def grab(pg, mode="RGB"):
    return np.array(Image.open(io.BytesIO(base64.b64decode(pg.evaluate(GRAB).split(",")[1]))).convert(mode))


def wait_done(pg, w):
    pg.wait_for_function(f"document.getElementById('status').textContent.includes('in your browser') && window.pixelforge.sr && window.pixelforge.sr.width === {w}", timeout=300000)


def main(url):
    net = load_model(DEFAULT_MODEL, torch.device("cpu"))
    d = build_eval_set("test", "real")
    tmp = Path(tempfile.mkdtemp())
    res = {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        pg = b.new_page(viewport={"width": 1280, "height": 1000}, accept_downloads=True)
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_function("document.getElementById('status').textContent.includes('in your browser')", timeout=180000)

        # 1) seamless
        lr = d["lr"][5]; Image.fromarray(lr).save(tmp / "tile.png")
        pg.check("#tileable")
        pg.set_input_files("#file", str(tmp / "tile.png")); wait_done(pg, 512)
        got = grab(pg)
        ref = upscale_rgb(net, lr, torch.device("cpu"), tileable=True)
        diff = np.abs(got.astype(int) - ref.astype(int))
        res["seamless"] = {"mean_abs_diff_vs_python": float(diff.mean()), "p99": float(np.percentile(diff, 99)),
                           "seam_browser": seam(got), "seam_python_tileable": seam(ref),
                           "seam_python_normal": seam(upscale_rgb(net, lr, torch.device("cpu")))}
        pg.uncheck("#tileable")

        # 2) transparency
        rgba = np.zeros((96, 96, 4), np.uint8); rgba[..., :3] = d["lr"][2][:96, :96]; rgba[..., :3][:24] = (255, 0, 255)
        rgba[24:72, 24:72, 3] = 255
        Image.fromarray(rgba, "RGBA").save(tmp / "decal.png")
        pg.set_input_files("#file", str(tmp / "decal.png")); wait_done(pg, 384)
        a = grab(pg, "RGBA")[..., 3]
        res["alpha"] = {"output_mode_rgba": True, "corner_alpha": int(a[5, 5]), "centre_alpha": int(a[192, 192]),
                        "opaque_fraction": float((a > 127).mean()), "expected_fraction": float((48 * 48) / (96 * 96))}

        # 3) batch
        files = []
        for k in range(3):
            f = tmp / f"b{k}.png"; Image.fromarray(d["lr"][10 + k]).save(f); files.append(str(f))
        pg.set_input_files("#file", files)
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Batch done')", timeout=300000)
        with pg.expect_download() as dl:
            pg.click("#dl")
        zp = tmp / "out.zip"; dl.value.save_as(zp)
        names = zipfile.ZipFile(zp).namelist()
        sizes = [Image.open(zipfile.ZipFile(zp).open(n)).size for n in names]
        res["batch"] = {"files_in": 3, "files_in_zip": len(names), "sizes": sizes}
        res["console_errors"] = errs
        b.close()
    out = ROOT / "eval" / "browser_check.json"
    data = json.loads(out.read_text()) if out.exists() else {}
    data["engine"] = res
    out.write_text(json.dumps(data, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
