"""Measured checks for the modding toolkit -> eval/toolkit_results.json + figures.

  python eval/toolkit_checks.py
1. chaiNNer compatibility: exported ESRGAN-format weights loaded by spandrel (the loader chaiNNer uses)
2. seamless tiling: wrap-edge seam on whole seamless test textures, with vs without --tileable
3. alpha halos: decals with junk colour under alpha=0, edge error with vs without colour bleeding
4. per-material gain: LPIPS improvement over bicubic by texture category (test set)
5. batch throughput of the CLI on a folder of test textures
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
import zlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.data import build_eval_set, load_manifest, read_rgb  # noqa: E402
from pixelforge.degrade import degrade  # noqa: E402
from pixelforge.toolkit import bleed_transparent, load_model, main as cli, upscale_rgb  # noqa: E402

FIG = ROOT / "assets" / "figures"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = ROOT / "models" / "rrdb_s_gan.pt"
SKIP_CATS = {"floor", "wall", "outdoor", "indoor", "man made", "natural", "clean", "dirty", "aerial", "rough"}


def chainner_check(net):
    import spandrel
    out = ROOT / "models" / "export" / "PixelForge_4x_RRDB-S.pth"
    cli(["export", str(out), "--model", str(MODEL)])
    try:
        desc = spandrel.ModelLoader().load_from_file(str(out))
    except Exception as e:  # record the incompatibility instead of failing the whole run
        msg = str(e).splitlines()
        return {"loader": f"spandrel {spandrel.__version__}", "compatible": False, "file": out.name,
                "reason": "spandrel detects ESRGAN but assumes 32 growth channels; PixelForge RRDB-S uses 16 "
                          "(" + next((l.strip() for l in msg if "size mismatch" in l), msg[0])[:160] + ")"}
    d = build_eval_set("test", "real")
    x = torch.from_numpy(d["lr"][0]).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad():
        a = desc.model.float().eval()(x).clamp(0, 1)
        b = net.float().cpu().eval()(x).clamp(0, 1)
    net.to(DEV)
    return {"loader": f"spandrel {spandrel.__version__}", "compatible": True, "detected_architecture": desc.architecture.name,
            "scale": desc.scale, "max_abs_pixel_diff": float((a - b).abs().max() * 255), "file": out.name}


def seam(img):
    """Mean colour jump across the wrap edges (right->left, bottom->top) minus the image's own
    neighbouring-pixel difference. ~0 = seamless; >0 = a visible seam when tiled."""
    f = img.astype(np.float32)
    wrap = (np.abs(f[:, -1] - f[:, 0]).mean() + np.abs(f[-1] - f[0]).mean()) / 2
    inner = (np.abs(np.diff(f, axis=1)).mean() + np.abs(np.diff(f, axis=0)).mean()) / 2
    return float(wrap - inner)


def tiling_check(net, n=30):
    res = {"hr": [], "normal": [], "tileable": []}
    examples = []
    for m in load_manifest("test")[:n]:
        hr = cv2.resize(read_rgb(ROOT / "data" / "polyhaven" / f"{m['slug']}.jpg"), (512, 512), interpolation=cv2.INTER_AREA)
        lr = degrade(hr, np.random.default_rng(zlib.crc32(m["slug"].encode())), 4, "real")
        a = upscale_rgb(net, lr, DEV, tile=192, tileable=False)
        b = upscale_rgb(net, lr, DEV, tile=192, tileable=True)
        res["hr"].append(seam(hr)); res["normal"].append(seam(a)); res["tileable"].append(seam(b))
        examples.append((res["normal"][-1] - res["tileable"][-1], m["slug"], a, b))
    summary = {k: float(np.mean(v)) for k, v in res.items()}
    summary["n"] = n
    summary["textures_with_visible_seam_normal"] = int(sum(s > 2.0 for s in res["normal"]))
    summary["textures_with_visible_seam_tileable"] = int(sum(s > 2.0 for s in res["tileable"]))
    # figure: the texture tiled 2x2, normal vs tileable (median-improvement example)
    examples.sort(key=lambda e: e[0])
    _, slug, a, b = examples[len(examples) // 2]
    # zoom on the point where four copies meet when the texture repeats in-game
    def seam_zoom(im):
        t = np.tile(im, (2, 2, 1))
        c = t.shape[0] // 2
        return cv2.resize(t[c - 96:c + 96, c - 96:c + 96], (480, 480), interpolation=cv2.INTER_NEAREST)
    fig = np.concatenate([seam_zoom(a), np.full((480, 16, 3), 255, np.uint8), seam_zoom(b)], 1)
    Image.fromarray(fig).save(FIG / "tileable_comparison.png")
    summary["figure_texture"] = slug
    return summary


def decal_mask(size, seed):
    rng = np.random.default_rng(seed)
    im = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(im)
    pts = [(size / 2 + np.cos(t) * size * rng.uniform(0.25, 0.42), size / 2 + np.sin(t) * size * rng.uniform(0.25, 0.42))
           for t in np.linspace(0, 2 * np.pi, 9)[:-1]]
    d.polygon(pts, fill=255)
    return np.array(im)


def alpha_check(net, n=20):
    d = build_eval_set("test", "real")
    errs = {"naive": [], "bleed": []}
    fig_rows = []
    for k in range(n):
        hr = d["hr"][k]
        a_hr = decal_mask(512, k)
        lr_rgb = cv2.resize(hr, (128, 128), interpolation=cv2.INTER_AREA)
        lr_a = cv2.resize(a_hr, (128, 128), interpolation=cv2.INTER_AREA)
        junk = lr_rgb.copy(); junk[lr_a == 0] = (255, 0, 255)          # what many assets hide under alpha=0
        ref = upscale_rgb(net, lr_rgb, DEV)                             # same model on the true colours
        naive = upscale_rgb(net, junk, DEV)
        fixed = upscale_rgb(net, bleed_transparent(junk, lr_a), DEV)
        a_up = cv2.resize(lr_a, (512, 512), interpolation=cv2.INTER_CUBIC)
        edge = (a_up > 127) & (cv2.dilate((a_up <= 127).astype(np.uint8), np.ones((17, 17), np.uint8)) > 0)
        errs["naive"].append(float(np.abs(naive.astype(int) - ref.astype(int))[edge].mean()))
        errs["bleed"].append(float(np.abs(fixed.astype(int) - ref.astype(int))[edge].mean()))
        if k == 3:
            def comp(rgb):  # composite on white so a halo is visible
                al = (a_up / 255.0)[..., None]
                return (rgb * al + 255 * (1 - al)).astype(np.uint8)
            fig_rows = [comp(naive), comp(fixed)]
    Image.fromarray(np.concatenate([fig_rows[0], np.full((512, 16, 3), 255, np.uint8), fig_rows[1]], 1)).save(FIG / "alpha_comparison.png")
    return {"n": n, "edge_error_naive": float(np.mean(errs["naive"])), "edge_error_bleed": float(np.mean(errs["bleed"])),
            "reduction_pct": float(100 * (1 - np.mean(errs["bleed"]) / np.mean(errs["naive"])))}


def category_gain():
    g = {r["slug"]: r for r in json.loads((ROOT / "eval" / "per_image" / "PixelForge_RRDB-S_GAN_real.json").read_text())}
    b = {r["slug"]: r for r in json.loads((ROOT / "eval" / "per_image" / "Bicubic_real.json").read_text())}
    by = defaultdict(list)
    for m in load_manifest("test"):
        cats = [c for c in m["categories"] if c not in SKIP_CATS] or ["other"]
        by[cats[0]].append(100 * (b[m["slug"]]["lpips"] - g[m["slug"]]["lpips"]) / b[m["slug"]]["lpips"])
    rows = sorted(({"category": c, "n": len(v), "lpips_gain_pct": float(np.mean(v))} for c, v in by.items() if len(v) >= 3),
                  key=lambda r: -r["lpips_gain_pct"])
    return rows


def throughput():
    d = build_eval_set("test", "real")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in"; src.mkdir()
        for k, s in enumerate(d["slug"]):
            Image.fromarray(d["lr"][k]).save(src / f"{s}.png")
        res = {}
        for dev in (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]:
            out = Path(tmp) / f"out_{dev}"
            files = src if dev == "cuda" else src  # cpu: same 60 files
            t0 = time.perf_counter()
            rep = cli(["upscale", str(files), str(out), "--device", dev, "--quiet"])
            res[dev] = {"textures": rep["done"], "seconds": round(time.perf_counter() - t0, 1),
                        "s_per_texture": round((time.perf_counter() - t0) / max(rep["done"], 1), 3)}
            shutil.rmtree(out, ignore_errors=True)
    return res


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    net = load_model(MODEL, DEV)
    out = {"chainner": chainner_check(net)}
    print("chaiNNer/spandrel:", out["chainner"], flush=True)
    out["tiling"] = tiling_check(net); print("tiling:", out["tiling"], flush=True)
    out["alpha"] = alpha_check(net); print("alpha:", out["alpha"], flush=True)
    out["categories"] = category_gain(); print("categories:", out["categories"], flush=True)
    out["throughput"] = throughput(); print("throughput:", out["throughput"], flush=True)
    (ROOT / "eval" / "toolkit_results.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
