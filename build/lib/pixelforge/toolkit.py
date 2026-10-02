"""PixelForge Engine: batch x4 upscaling of game texture folders.

  pixelforge upscale <in_dir> <out_dir> [--tileable] [--model models/rrdb_s_gan.pt]
  pixelforge export  <out.pth>  [--model models/rrdb_s_gan.pt]   # ESRGAN-format weights + model card

Features
- folder in -> folder out, sub-folders and file names kept (output is always PNG)
- alpha: transparent pixels' colours are filled from opaque neighbours before upscaling (stops colour
  halos/fringes), alpha itself is upscaled separately and recombined
- tileable: wrap-around padding so seamless textures stay seamless after upscaling
- tiled inference with overlap, so any texture size fits in memory
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .models import build_generator

DEFAULT_MODEL = Path(__file__).resolve().parent / "weights" / "pixelforge_rrdb_s_gan.pt"
EXTS = {".png", ".jpg", ".jpeg", ".tga", ".bmp", ".webp", ".dds", ".tif", ".tiff"}
SCALE = 4


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "config" in ck:  # PixelForge checkpoint
        net = build_generator(**ck["config"])
        net.load_state_dict(ck["ema"])
    else:  # ESRGAN-format .pth exported by `export`
        meta = ck.get("pixelforge_config", {"arch": "rrdb", "nf": 32, "nb": 6, "gc": 16})
        net = build_generator(**meta)
        net.load_state_dict(ck.get("params_ema", ck))
    return net.to(device).eval()


@torch.no_grad()
def _run(net, img: np.ndarray, device) -> np.ndarray:
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().div(255).to(device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        y = net(x)
    return (y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def upscale_rgb(net, img: np.ndarray, device, tile=192, overlap=16, tileable=False) -> np.ndarray:
    """x4 upscale an HxWx3 uint8 image in overlapping tiles; optional wrap-around for seamless textures."""
    pad = overlap if tileable else 0
    src = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="wrap") if pad else img
    H, W = src.shape[:2]
    out = np.zeros((H * SCALE, W * SCALE, 3), np.uint8)
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            y0, x0 = max(ty - overlap, 0), max(tx - overlap, 0)
            y1, x1 = min(ty + tile + overlap, H), min(tx + tile + overlap, W)
            sr = _run(net, np.ascontiguousarray(src[y0:y1, x0:x1]), device)
            cy, cx = (ty - y0) * SCALE, (tx - x0) * SCALE
            h, w = min(tile, H - ty) * SCALE, min(tile, W - tx) * SCALE
            out[ty * SCALE:ty * SCALE + h, tx * SCALE:tx * SCALE + w] = sr[cy:cy + h, cx:cx + w]
    if pad:
        out = out[pad * SCALE:-pad * SCALE, pad * SCALE:-pad * SCALE]
    return out


def bleed_transparent(rgb: np.ndarray, alpha: np.ndarray, sigma=4.0, rounds=4) -> np.ndarray:
    """Replace colours under fully transparent pixels with a blur of nearby opaque colours.
    Game assets often hide junk colours (magenta, black) under alpha=0; without this the network
    smears them into visible fringes around the edges."""
    mask = (alpha > 0).astype(np.float32)
    if mask.all() or not mask.any():
        return rgb
    out = rgb.astype(np.float32)
    known = mask.copy()
    for _ in range(rounds):  # repeated normalised blur grows the opaque colours outwards
        num = cv2.GaussianBlur(out * known[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(known, (0, 0), sigma)[..., None]
        fill = num / np.maximum(den, 1e-6)
        out = np.where(known[..., None] > 0, out, fill)
        known = np.maximum(known, (den[..., 0] > 1e-3).astype(np.float32))
    return np.clip(out, 0, 255).round().astype(np.uint8)


def upscale_image(net, im: Image.Image, device, tileable=False, tile=192, bleed=True) -> Image.Image:
    has_alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
    if has_alpha:
        rgba = np.array(im.convert("RGBA"))
        rgb, alpha = rgba[..., :3], rgba[..., 3]
        if bleed:
            rgb = bleed_transparent(rgb, alpha)
        sr = upscale_rgb(net, rgb, device, tile=tile, tileable=tileable)
        a = cv2.resize(alpha, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(np.dstack([sr, a]), "RGBA")
    sr = upscale_rgb(net, np.array(im.convert("RGB")), device, tile=tile, tileable=tileable)
    return Image.fromarray(sr, "RGB")


def cmd_upscale(args):
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_model(args.model, device)
    src, dst = Path(args.input), Path(args.output)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS) if src.is_dir() else [src]
    report = {"done": 0, "skipped": 0, "failed": [], "seconds": 0.0, "device": str(device), "tileable": args.tileable}
    t0 = time.perf_counter()
    for i, f in enumerate(files, 1):
        rel = f.relative_to(src) if src.is_dir() else Path(f.name)
        out = (dst / rel).with_suffix(".png")
        if out.exists() and not args.overwrite:
            report["skipped"] += 1
            continue
        try:
            img = upscale_image(net, Image.open(f), device, tileable=args.tileable, tile=args.tile)
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(out)
            report["done"] += 1
        except Exception as e:  # keep going; list failures at the end
            report["failed"].append({"file": str(rel), "error": f"{type(e).__name__}: {e}"})
        if not args.quiet:
            print(f"[{i}/{len(files)}] {rel}", flush=True)
    report["seconds"] = round(time.perf_counter() - t0, 2)
    print(json.dumps(report, indent=2))
    return report


def cmd_export(args):
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # ESRGAN / Real-ESRGAN layout ("params_ema" + official layer names) so tools that load
    # ESRGAN-family models (e.g. chaiNNer via spandrel) can detect and run it.
    torch.save({"params_ema": ck["ema"], "pixelforge_config": ck["config"]}, out)
    card = out.with_suffix(".md")
    card.write_text(f"""# {out.stem}

- **Task:** 4x super-resolution of game textures (albedo / colour maps)
- **Architecture:** ESRGAN RRDBNet, {ck['config']['nb']} RRDB blocks, {ck['config']['nf']} features, grow {ck['config']['gc']} ({ck.get('params', 0) / 1e6:.2f}M params)
- **Training data:** Poly Haven textures (CC0), realistic degradations (blur, downscale, noise, JPEG); L1 then GAN stage
- **Licence:** PolyForm Noncommercial 1.0.0 (free for modding and personal use; commercial licence on request)
- **Source:** https://github.com/Priyanshu-Singh-git/pixelforge
- **Not for:** pixel-art sprites, normal maps, text-heavy UI textures
""", encoding="utf-8")
    print(f"wrote {out} and {card}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pixelforge", description="PixelForge Engine: x4 game texture upscaling for modders")
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("upscale", help="x4 upscale a texture or a folder of textures")
    up.add_argument("input"); up.add_argument("output")
    up.add_argument("--model", default=str(DEFAULT_MODEL))
    up.add_argument("--tileable", action="store_true", help="keep seamless textures seamless (wrap-around padding)")
    up.add_argument("--tile", type=int, default=192, help="tile size in input pixels (lower = less memory)")
    up.add_argument("--device", default="auto")
    up.add_argument("--overwrite", action="store_true")
    up.add_argument("--quiet", action="store_true")
    ex = sub.add_parser("export", help="write ESRGAN-format weights (+ model card) for other tools")
    ex.add_argument("output")
    ex.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args(argv)
    return cmd_upscale(args) if args.cmd == "upscale" else cmd_export(args)


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
