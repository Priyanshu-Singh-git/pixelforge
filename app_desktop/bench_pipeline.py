"""Headless end-to-end FPS of the PixelForge Live pipeline: synthetic frame -> upscale -> resize-to-screen.
Measures the real per-frame cost (minus actual screen capture, which runs on its own thread and only caps at
the game's own FPS). Writes app_desktop/bench.json.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app_desktop"))


def main():
    import onnxruntime as ort
    out = {"provider": None, "runs": []}
    models = {"live": ".cache/rt_live_fp16.onnx", "lite": ".cache/rt_lite_fp16.onnx"}
    screen = (1920, 1080)
    for name, mp in models.items():
        if not (ROOT / mp).exists():
            continue
        providers = [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        s = ort.InferenceSession(str(ROOT / mp), providers=providers)
        out["provider"] = s.get_providers()[0]
        inp = s.get_inputs()[0].name
        for (w, h), lbl in [((960, 540), "540p->1080p"), ((640, 360), "360p->720p")]:
            frame = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
            def step():
                x = frame.transpose(2, 0, 1)[None].astype(np.float16) / 255.0
                y = s.run(None, {inp: x})[0]
                o = np.clip(y[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
                cv2.resize(o, screen, interpolation=cv2.INTER_LINEAR)  # display scale
            for _ in range(5):
                step()
            t = time.perf_counter(); n = 60
            for _ in range(n):
                step()
            ms = (time.perf_counter() - t) / n * 1000
            out["runs"].append({"model": name, "res": lbl, "ms": round(ms, 2), "fps": round(1000 / ms)})
            print(f"{name:5s} {lbl}: {ms:5.1f} ms end-to-end = {1000/ms:3.0f} FPS")
    (ROOT / "app_desktop" / "bench.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
