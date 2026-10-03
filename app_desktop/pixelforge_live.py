"""PixelForge Live - desktop AI upscaler (Lossless-Scaling-style, powered by the PixelForge model).

Captures a window/region, upscales each frame with the PixelForge SRVGG model on the GPU (ONNX DirectML),
and shows the result borderless-fullscreen. Capture -> upscale -> display; nothing is injected into the game,
so it is safe for single-player / offline use and trips no anti-cheat.

  python app_desktop/pixelforge_live.py            # default: live model, 2x, auto region
  python app_desktop/pixelforge_live.py --model lite --scale 2 --region 960x540

Hotkeys:  F  toggle fullscreen    C  cycle compare (AI | bilinear | off-split)    H  toggle HUD    ESC quit
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"live": "pixelforge_live_x2.onnx", "lite": "pixelforge_lite_x2.onnx"}  # baked uint8->uint8 graphs


class Upscaler:
    """ONNX SRVGG x2 on the GPU via DirectML (fp16), with a bilinear fallback for comparison."""

    def __init__(self, model_path: Path, fp16=True):
        import onnxruntime as ort
        providers = [p for p in ("DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in ort.get_available_providers()]
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(model_path), so, providers=providers)
        self.provider = self.sess.get_providers()[0]
        self.dtype = np.float16 if fp16 else np.float32
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, rgb_u8: np.ndarray) -> np.ndarray:
        # frame in / frame out are both uint8 HWC; normalize + denorm happen on the GPU in the graph
        return self.sess.run(None, {self.name: np.ascontiguousarray(rgb_u8)})[0]


class Capture:
    """Background grab of a screen region into a shared latest-frame buffer."""

    def __init__(self, region=None):
        import dxcam
        self.cam = dxcam.create(output_color="RGB")
        self.region = region  # (l, t, r, b) or None = full primary
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.grabs = 0

    def _loop(self):
        self.cam.start(target_fps=0, region=self.region)
        while self.running:
            f = self.cam.get_latest_frame()
            if f is not None:
                with self.lock:
                    self.frame = f
                    self.grabs += 1
        self.cam.stop()

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def latest(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False


def parse_region(s):
    if not s:
        return None
    if "+" in s:  # WxH+X+Y
        wh, x, y = s.split("+"); w, h = map(int, wh.split("x")); x, y = int(x), int(y)
    else:         # WxH centred on primary
        import ctypes
        user = ctypes.windll.user32
        sw, sh = user.GetSystemMetrics(0), user.GetSystemMetrics(1)
        w, h = map(int, s.split("x")); x, y = (sw - w) // 2, (sh - h) // 2
    return (x, y, x + w, y + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="live", choices=list(MODELS))
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--region", default="960x540", help="WxH or WxH+X+Y of the game area to capture")
    ap.add_argument("--models-dir", default=str(ROOT / "app_desktop" / "models"))
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    region = parse_region(args.region)
    up = Upscaler(Path(args.models_dir) / MODELS[args.model], fp16=not args.fp32)
    cap = Capture(region)
    cap.start()
    time.sleep(0.3)

    import pygame
    pygame.init()
    info = pygame.display.Info()
    screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.FULLSCREEN | pygame.SCALED)
    pygame.display.set_caption("PixelForge Live")
    font = pygame.font.SysFont("Consolas", 22)
    clock = pygame.time.Clock()

    mode, show_hud, running = 0, True, True   # mode: 0 AI, 1 bilinear, 2 split
    ema_fps, ema_up = 0.0, 0.0
    while running:
        t0 = time.perf_counter()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_c:
                    mode = (mode + 1) % 3
                elif e.key == pygame.K_h:
                    show_hud = not show_hud
        frame = cap.latest()
        if frame is None:
            clock.tick(60); continue
        import cv2
        tu = time.perf_counter()
        if mode == 1:
            out = cv2.resize(frame, (frame.shape[1] * args.scale, frame.shape[0] * args.scale), interpolation=cv2.INTER_LINEAR)
        else:
            out = up(frame)
            if mode == 2:  # left half AI, right half bilinear, to compare
                bil = cv2.resize(frame, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
                out[:, out.shape[1] // 2:] = bil[:, out.shape[1] // 2:]
        up_ms = (time.perf_counter() - tu) * 1000

        surf = pygame.image.frombuffer(np.ascontiguousarray(out).tobytes(), (out.shape[1], out.shape[0]), "RGB")
        surf = pygame.transform.smoothscale(surf, (info.current_w, info.current_h))
        screen.blit(surf, (0, 0))

        dt = time.perf_counter() - t0
        ema_fps = 0.9 * ema_fps + 0.1 * (1 / max(dt, 1e-4))
        ema_up = 0.9 * ema_up + 0.1 * up_ms
        if show_hud:
            lines = [f"PixelForge Live  [{up.provider.replace('ExecutionProvider','')}]  {['AI','Bilinear','AI|Bil split'][mode]}",
                     f"{ema_fps:4.0f} FPS   upscale {ema_up:4.1f} ms   {frame.shape[1]}x{frame.shape[0]} -> {out.shape[1]}x{out.shape[0]}",
                     "F fullscreen  C compare  H hud  ESC quit"]
            for i, ln in enumerate(lines):
                px = font.render(ln, True, (233, 255, 20))
                screen.blit(pygame.Surface(px.get_size()).convert_alpha(), (14, 12 + i * 26))
                screen.blit(px, (14, 12 + i * 26))
        pygame.display.flip()
        clock.tick(240)

    cap.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
