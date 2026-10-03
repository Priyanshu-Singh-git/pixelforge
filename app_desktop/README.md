# PixelForge Live

Real-time AI game upscaler for the desktop (Lossless-Scaling-style), powered by the PixelForge model.
Captures a game window, upscales each frame 2x on your GPU (ONNX Runtime DirectML), and shows the result
borderless-fullscreen. Capture -> upscale -> display; **nothing is injected into the game**, so it is safe for
single-player / offline use and does not interact with anti-cheat.

## Run

```bash
pip install -r app_desktop/requirements.txt
pip install onnxruntime-directml                 # GPU upscaling on any Windows GPU
python app_desktop/pixelforge_live.py            # defaults: lite model, capture 960x540 centred, 2x
python app_desktop/pixelforge_live.py --model live --region 960x540+100+100
```

Hotkeys: `F` fullscreen · `C` compare (AI / bilinear / split) · `H` toggle HUD · `ESC` quit.

## Measured performance (RTX 3050 laptop, DirectML, real game-like video)

| Model | 540p -> 1080p | 360p -> 720p |
|---|---|---|
| Lite (nf8) | **65 FPS** | 109 FPS |
| Live / quality (nf16) | 50 FPS | 82 FPS |

The models bake uint8 -> normalize -> upscale -> uint8 into the ONNX graph so the whole step runs on the GPU
in one call (numpy pre/post on 1080p frames would cost ~300 ms/frame otherwise).

## How it works vs DLSS

This is **capture-based**: you play through the app's fullscreen window. The game still renders at its own
resolution, so this sharpens/enlarges the output — it does **not** give DLSS's "render low, get free FPS"
performance win, which requires engine-level integration (motion vectors, depth). A sanctioned engine/emulator
plugin would be the route to true render-time upscaling (roadmap), not injection.

Models: `app_desktop/models/pixelforge_{lite,live}_x2.onnx`, exported by `app_desktop/export_live_models.py`
from the real-time checkpoints (`models/rt_{lite,live}_x2.pt`, trained by `train/run_realtime.sh`).
