"""Build the PixelForge deck: measured results -> deck.pptx -> 1920x1080 slide PNGs + PDF.

  python presentation/build_deck.py --model rrdb_s_gan
Every number on a slide is computed here from eval/results.jsonl and eval/browser_check.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "presentation"
FIG = ROOT / "assets" / "figures"
DEMO = "https://priyanshu-singh-git.github.io/pixelforge/"
REPO = "https://github.com/Priyanshu-Singh-git/pixelforge"
LABEL = {"rrdb_s_gan": "PixelForge RRDB-S, GAN (mine)", "rrdb_s_psnr": "PixelForge RRDB-S, L1 (mine)",
         "srvgg_s_psnr": "PixelForge SRVGG-S, L1 (mine)"}


def rounded(src: Path, radius_frac=0.045) -> str:
    """Copy of an image with transparent rounded corners (curved boundaries on every deck image)."""
    im = Image.open(src).convert("RGBA")
    r = int(min(im.size) * radius_frac)
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, im.width - 1, im.height - 1], r, fill=255)
    im.putalpha(mask)
    dst = OUT / "rounded" / src.name
    dst.parent.mkdir(exist_ok=True)
    im.save(dst)
    return str(dst)


def qr(url):
    import qrcode
    q = qrcode.QRCode(border=2, box_size=12)
    q.add_data(url); q.make(fit=True)
    q.make_image(fill_color="#14532d", back_color="white").convert("RGB").save(OUT / "qr.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rrdb_s_gan")
    ap.add_argument("--video-url", default=None)
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "eval" / "results.jsonl").read_text().splitlines()]
    real = {r["method"]: r for r in rows if r["condition"] == "real"}
    mine, bic = real[LABEL[args.model]], real["Bicubic"]
    ref = real["Real-ESRGAN x4plus (pretrained)"]
    l1, sv = real[LABEL["rrdb_s_psnr"]], real[LABEL["srvgg_s_psnr"]]
    browser = json.loads((ROOT / "eval" / "browser_check.json").read_text())
    browser_s = float(np.median([s["ms"] for s in browser["samples"]])) / 1000
    n_train = sum(m["split"] == "train" for m in json.loads((ROOT / "data" / "manifest.json").read_text()))

    lp_gain = round(100 * (bic["lpips"] - mine["lpips"]) / bic["lpips"])
    size_ratio = round(ref["params_m"] / mine["params_m"])
    stats = [
        [f"{lp_gain}%", "closer to the real HD look than bicubic", f"LPIPS perceptual distance, {mine['n']} unseen game textures"],
        [f"{size_ratio}×", f"smaller than Real-ESRGAN ({mine['params_m']:.1f}M vs {ref['params_m']:.1f}M parameters)",
         f"LPIPS {mine['lpips']:.3f} vs {ref['lpips']:.3f} (lower is better)"],
        [f"{browser_s:.1f} s", "to upscale a texture 4× in a laptop browser", "128×128 → 512×512, no GPU, no upload"],
    ]
    order = ["Bicubic", "Lanczos", "Real-ESRGAN x4plus (pretrained)", "Real-ESRGAN general-v3 (pretrained)",
             LABEL["srvgg_s_psnr"], LABEL["rrdb_s_psnr"], LABEL["rrdb_s_gan"]]
    table = [{"method": m.replace(" (mine)", " ★"), "params": f"{real[m]['params_m']:.2f}M" if real[m]["params_m"] else "–",
              "psnr": real[m]["psnr"], "ssim": real[m]["ssim"], "lpips": real[m]["lpips"],
              "gpu_ms": f"{real[m]['gpu_ms']:.1f}" if real[m].get("gpu_ms") else "–", "mine": "(mine)" in m}
             for m in order if m in real]
    findings = [
        (f"Architecture: the {l1['params_m'] / sv['params_m']:.0f}× smaller SRVGG-S ({sv['params_m']:.2f}M) gets within "
         f"{l1['psnr'] - sv['psnr']:.2f} dB PSNR of RRDB-S ({l1['params_m']:.2f}M) and "
         + ("slightly better" if sv["lpips"] < l1["lpips"] else "slightly worse")
         + f" LPIPS ({sv['lpips']:.3f} vs {l1['lpips']:.3f}), at {sv['gpu_ms']:.0f} vs {l1['gpu_ms']:.0f} ms: a strong candidate for real-time use."),
        f"L1 vs GAN: the GAN stage cuts LPIPS from {l1['lpips']:.3f} to {mine['lpips']:.3f} (looks far more real) at a cost of "
        f"{l1['psnr'] - mine['psnr']:.2f} dB PSNR, the classic fidelity vs realism trade-off.",
        f"Versus Real-ESRGAN x4plus (trained on photos): LPIPS {mine['lpips']:.3f} vs {ref['lpips']:.3f}, PSNR "
        f"{mine['psnr']:.2f} vs {ref['psnr']:.2f} dB, with {size_ratio}× fewer parameters.",
        "Single training run per model; synthetic degradations; pixel-art sprites and normal maps not yet handled.",
    ]
    tk = json.loads((ROOT / "eval" / "toolkit_results.json").read_text())
    engine = {
        "seam_normal": tk["tiling"]["textures_with_visible_seam_normal"], "seam_tileable": tk["tiling"]["textures_with_visible_seam_tileable"],
        "seam_n": tk["tiling"]["n"], "alpha_before": tk["alpha"]["edge_error_naive"], "alpha_after": tk["alpha"]["edge_error_bleed"],
        "alpha_cut": round(tk["alpha"]["reduction_pct"]), "gpu_s": tk["throughput"]["cuda"]["s_per_texture"],
        "cpu_s": tk["throughput"]["cpu"]["s_per_texture"], "batch_n": tk["throughput"]["cpu"]["textures"],
    }
    zoom = Image.open(FIG / "zoom_comparison.png")
    zoom.crop((0, 0, zoom.width, zoom.height // 3)).save(FIG / "zoom_row.png")
    qr(DEMO)
    data = {
        "demo_url": DEMO, "repo_url": REPO, "video_url": args.video_url, "stats": stats, "table": table,
        "findings": findings, "engine": engine, "n_test": mine["n"], "n_train": n_train, "params_m": mine["params_m"],
        "ref_params_m": ref["params_m"], "browser_s": browser_s,
        "assets": {"hero": rounded(FIG / "hero_grid.png", 0.035), "shot": rounded(FIG / "demo_screenshot.png", 0.03),
                   "zoom": rounded(FIG / "zoom_comparison.png", 0.03), "zoom_row": rounded(FIG / "zoom_row.png", 0.08),
                   "qr": str(OUT / "qr.png"), "tileable": rounded(FIG / "tileable_comparison.png", 0.04),
                   "alpha": rounded(FIG / "alpha_comparison.png", 0.04)},
    }
    (OUT / "deck_data.json").write_text(json.dumps(data, indent=2))
    subprocess.run(["node", "build_pptx.js"], check=True, cwd=OUT)
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(OUT / "export_slides.ps1")],
                   check=True, cwd=OUT)


if __name__ == "__main__":
    main()
