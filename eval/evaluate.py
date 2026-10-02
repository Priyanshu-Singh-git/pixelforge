"""Evaluate x4 upscalers on the held-out TEST textures (never used for training or checkpoint choice).

Two input conditions per texture, fixed forever (see pixelforge.data.build_eval_set):
  real    - blur + random downscale + noise + JPEG (what old/compressed game assets look like)
  bicubic - clean bicubic downscale (classic SR benchmark setting)
Metrics: PSNR / SSIM (fidelity, higher better), LPIPS-Alex (perceptual, lower better),
GPU and CPU time per 128x128 -> 512x512 texture, parameter count.

  python eval/evaluate.py                      # all methods with available weights
Writes eval/results.jsonl (one row per method x condition) and eval/per_image/<method>_<cond>.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.data import build_eval_set  # noqa: E402
from pixelforge.metrics import lpips_dist, psnr, ssim  # noqa: E402
from pixelforge.models import RRDBNet, SRVGGNetCompact, build_generator, n_params  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def classical(flag):
    return lambda lr: cv2.resize(lr, (lr.shape[1] * 4, lr.shape[0] * 4), interpolation=flag)


def load_pretrained(name):
    if name == "RealESRGAN_x4plus":
        net, key = RRDBNet(num_feat=64, num_block=23), "params_ema"
    else:
        net, key = SRVGGNetCompact(num_feat=64, num_conv=32), "params"
    sd = torch.load(ROOT / "models" / "pretrained" / f"{name}.pth", map_location="cpu", weights_only=False)
    net.load_state_dict(sd.get(key, sd))
    return net


def load_ours(name):
    ck = torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    net = build_generator(**ck["config"])
    net.load_state_dict(ck["ema"])
    return net


def neural(net):
    net = net.to(DEV).eval()

    @torch.no_grad()
    def run(lr):
        x = torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255).to(DEV)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEV.type == "cuda"):
            y = net(x)
        return (y.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    return run


def time_it(fn, lr, n=10):
    fn(lr)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(lr)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t.append(1000 * (time.perf_counter() - t0))
    return float(np.median(t))


@torch.no_grad()
def cpu_ms(net, lr, threads=4, n=3):
    torch.set_num_threads(threads)
    net = net.cpu().float().eval()
    x = torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255)
    net(x)
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); net(x); t.append(1000 * (time.perf_counter() - t0))
    return float(np.median(t))


def methods():
    yield "Nearest neighbour", "classical", classical(cv2.INTER_NEAREST), None
    yield "Bicubic", "classical", classical(cv2.INTER_CUBIC), None
    yield "Lanczos", "classical", classical(cv2.INTER_LANCZOS4), None
    for name, label in [("RealESRGAN_x4plus", "Real-ESRGAN x4plus (pretrained)"),
                        ("realesr-general-x4v3", "Real-ESRGAN general-v3 (pretrained)")]:
        if (ROOT / "models" / "pretrained" / f"{name}.pth").exists():
            yield label, "pretrained", None, (load_pretrained, name)
    for name, label in [("rrdb_s_psnr", "PixelForge RRDB-S, L1 (mine)"),
                        ("srvgg_s_psnr", "PixelForge SRVGG-S, L1 (mine)"),
                        ("rrdb_s_gan", "PixelForge RRDB-S, GAN (mine)")]:
        if (ROOT / "models" / f"{name}.pt").exists():
            yield label, "mine", None, (load_ours, name)


def main():
    out = ROOT / "eval" / "results.jsonl"
    out.unlink(missing_ok=True)
    (ROOT / "eval" / "per_image").mkdir(parents=True, exist_ok=True)
    sets = {c: build_eval_set("test", c) for c in ("real", "bicubic")}
    for label, kind, fn, loader in methods():
        net = None
        if loader:
            net = loader[0](loader[1])
            fn = neural(net)
        for cond, d in sets.items():
            rows = []
            for k in range(len(d["lr"])):
                sr = fn(d["lr"][k])
                rows.append({"slug": str(d["slug"][k]), "psnr": psnr(sr, d["hr"][k]),
                             "ssim": ssim(sr, d["hr"][k]), "lpips": lpips_dist(sr, d["hr"][k], DEV)})
            key = label.split(" (")[0].replace(" ", "_").replace(",", "")
            (ROOT / "eval" / "per_image" / f"{key}_{cond}.json").write_text(json.dumps(rows))
            res = {"method": label, "kind": kind, "condition": cond, "n": len(rows),
                   "psnr": float(np.mean([r["psnr"] for r in rows])),
                   "ssim": float(np.mean([r["ssim"] for r in rows])),
                   "lpips": float(np.mean([r["lpips"] for r in rows])),
                   "params_m": (n_params(net) / 1e6) if net is not None else 0.0,
                   "gpu_ms": time_it(fn, d["lr"][0]) if net is not None else None,
                   "gpu": torch.cuda.get_device_name(0) if DEV.type == "cuda" else None}
            if net is not None and cond == "real":
                res["cpu_ms_4threads"] = cpu_ms(net, d["lr"][0])
                net.to(DEV)
            with open(out, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"{label:40s} {cond:8s} PSNR {res['psnr']:.2f}  SSIM {res['ssim']:.4f}  LPIPS {res['lpips']:.4f}"
                  + (f"  {res['gpu_ms']:.1f} ms GPU" if res["gpu_ms"] else ""), flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
