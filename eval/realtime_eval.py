"""Real-time 2x models: quality on held-out textures + per-frame speed, ONNX export for the Live page.

  python eval/realtime_eval.py
Writes eval/realtime_results.json, docs/models/pixelforge_live_x2.onnx / pixelforge_lite_x2.onnx, docs/live_meta.json
"""
from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.data import build_eval_set  # noqa: E402
from pixelforge.metrics import lpips_dist, psnr  # noqa: E402
from pixelforge.models import build_generator, n_params  # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {"live": "rt_live_x2", "lite": "rt_lite_x2"}
RES = {"p360": (640, 360), "p540": (960, 540)}


def load(name):
    ck = torch.load(ROOT / "models" / f"{name}.pt", map_location="cpu", weights_only=False)
    net = build_generator(**ck["config"]).eval()
    net.load_state_dict(ck["ema"])
    return net


def quality(net):
    d = build_eval_set("test", "frame", scale=2)
    rows = {"model": [], "bilinear": [], "bicubic": [], "lanczos": []}
    net = net.to(DEV)
    for lr, hr in zip(d["lr"], d["hr"]):
        with torch.no_grad():
            y = net(torch.from_numpy(lr).permute(2, 0, 1)[None].float().div(255).to(DEV)).clamp(0, 1)
        outs = {"model": (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)}
        for k, f in (("bilinear", cv2.INTER_LINEAR), ("bicubic", cv2.INTER_CUBIC), ("lanczos", cv2.INTER_LANCZOS4)):
            outs[k] = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=f)
        for k, o in outs.items():
            rows[k].append((psnr(o, hr), lpips_dist(o, hr, DEV)))
    net.cpu()
    return {k: {"psnr": float(np.mean([a for a, _ in v])), "lpips": float(np.mean([b for _, b in v]))} for k, v in rows.items()}


def export(net, path):
    torch.onnx.export(net, torch.rand(1, 3, 64, 64), str(path), opset_version=17, input_names=["lr"], output_names=["sr"],
                      dynamic_axes={"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}})


def cpu_speed(path, threads, w, h, n=8):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.rand(1, 3, h, w).astype(np.float32)
    s.run(None, {"lr": x}); s.run(None, {"lr": x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter(); s.run(None, {"lr": x}); t.append(1000 * (time.perf_counter() - t0))
    return float(np.median(t))


@torch.no_grad()
def gpu_speed(net, w, h, n=50):
    m = net.to("cuda").half().eval()
    x = torch.rand(1, 3, h, w, device="cuda").half()
    for _ in range(5): m(x)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): m(x)
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / n
    net.float().cpu()
    return ms


def main():
    out = {"cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "models": {}}
    (ROOT / "docs" / "models").mkdir(parents=True, exist_ok=True)
    for key, name in MODELS.items():
        net = load(name)
        onnx_path = ROOT / "docs" / "models" / f"pixelforge_{key}_x2.onnx"
        export(net, onnx_path)
        r = {"params": n_params(net), "onnx": onnx_path.name, "quality": quality(net), "speed_ms": {}}
        for rk, (w, h) in RES.items():
            for th in (1, 2, 4):
                r["speed_ms"][f"cpu{th}_{rk}"] = cpu_speed(onnx_path, th, w, h)
            if torch.cuda.is_available():
                r["speed_ms"][f"gpu_{rk}"] = gpu_speed(net, w, h)
        out["models"][key] = r
        print(key, json.dumps(r), flush=True)
    (ROOT / "eval" / "realtime_results.json").write_text(json.dumps(out, indent=2))

    fps = lambda ms: f"{1000 / ms:.0f} fps"  # noqa: E731
    sp = lambda k, rk: out["models"][k]["speed_ms"][rk]  # noqa: E731
    speed = [{"device": f"GPU, Live model ({out['gpu']})", "p360": fps(sp("live", "gpu_p360")), "p540": fps(sp("live", "gpu_p540"))},
             {"device": "No GPU, 4 CPU threads, Lite", "p360": fps(sp("lite", "cpu4_p360")), "p540": fps(sp("lite", "cpu4_p540"))},
             {"device": "No GPU, 2 CPU threads, Lite (low-end / phone-class)", "p360": fps(sp("lite", "cpu2_p360")), "p540": fps(sp("lite", "cpu2_p540"))},
             {"device": "No GPU, 4 CPU threads, Live", "p360": fps(sp("live", "cpu4_p360")), "p540": fps(sp("live", "cpu4_p540"))}]
    meta = {"models": {k: {"file": v["onnx"], "params": v["params"]} for k, v in out["models"].items()}, "speed": speed,
            "speed_note": f"Model time only, measured on a laptop ({out['cpu']}, {out['gpu']}) with ONNX Runtime / PyTorch. "
                          "Your browser's fps is shown live above. Not yet measured on a real phone.",
            "demo": [f"samples/{p.name}" for p in sorted((ROOT / "docs" / "samples").glob("*_hr.jpg"))]}
    (ROOT / "docs" / "live_meta.json").write_text(json.dumps(meta, indent=2))
    print("wrote docs/live_meta.json")


if __name__ == "__main__":
    main()
