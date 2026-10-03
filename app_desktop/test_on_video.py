"""Run a low-res video through PixelForge Live: measure real FPS, write an upscaled MP4 and a
before/after comparison PNG. Proves the desktop pipeline on moving game-like content.

  python app_desktop/test_on_video.py --video .cache/test_game_540p.mp4 --model lite
Writes assets/figures/live_compare.png, assets/figures/live_upscaled.mp4, eval/live_results.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--model", default="lite", choices=["live", "lite"])
    ap.add_argument("--frames", type=int, default=240)
    args = ap.parse_args()

    import onnxruntime as ort
    mp = ROOT / "app_desktop" / "models" / f"pixelforge_{args.model}_x2.onnx"
    providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    s = ort.InferenceSession(str(mp), providers=providers)
    provider = s.get_providers()[0]
    name = s.get_inputs()[0].name

    cap = cv2.VideoCapture(args.video)
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    import subprocess
    writer = None
    times, n = [], 0
    compare = None
    while n < args.frames:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = bgr[..., ::-1]
        t = time.perf_counter()
        out = s.run(None, {name: np.ascontiguousarray(rgb)})[0]   # uint8 HWC, 2x
        times.append(time.perf_counter() - t)
        if writer is None:
            Hh, Ww = out.shape[:2]
            writer = subprocess.Popen([ff, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Ww}x{Hh}", "-r", str(fps_in), "-i", "-",
                                       "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(ROOT / "assets/figures/live_upscaled.mp4")],
                                      stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        writer.stdin.write(np.ascontiguousarray(out[..., ::-1]).tobytes())
        if n == 120:   # a mid clip frame for the comparison
            bil = cv2.resize(rgb, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
            crop = lambda im: im[out.shape[0] // 3:out.shape[0] // 3 + 260, out.shape[1] // 2 - 130:out.shape[1] // 2 + 130]
            lab = lambda im, t, c: cv2.putText(im.copy(), t, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2)
            compare = np.concatenate([lab(crop(bil), "Bilinear", (40, 40, 220)), np.full((260, 8, 3), 255, np.uint8),
                                      lab(crop(out), "PixelForge", (20, 180, 20))], 1)
        n += 1
    cap.release()
    if writer:
        writer.stdin.close(); writer.wait()
    if compare is not None:
        cv2.imwrite(str(ROOT / "assets/figures/live_compare.png"), compare[..., ::-1])

    times.sort()
    med = times[len(times) // 2]
    res = {"model": args.model, "provider": provider, "frames": n, "median_ms": round(med * 1000, 2),
           "fps": round(1 / med), "p95_ms": round(times[int(len(times) * 0.95)] * 1000, 2),
           "input_res": f"{rgb.shape[1]}x{rgb.shape[0]}", "output_res": f"{out.shape[1]}x{out.shape[0]}"}
    (ROOT / "eval" / "live_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
