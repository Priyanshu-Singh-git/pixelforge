"""Export trained real-time models to GPU-ready ONNX for the desktop app (PixelForge Live).

Bakes uint8-HWC -> upscale -> uint8-HWC into the graph so the whole step runs on the GPU in one call
(this is what makes the pipeline real-time; numpy pre/post on 1080p frames is ~300 ms otherwise).

  python app_desktop/export_live_models.py
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.models import build_generator  # noqa: E402

CKPTS = {"live": ("rt_live_x2.pt", 16, 4), "lite": ("rt_lite_x2.pt", 8, 3)}
OUT = ROOT / "app_desktop" / "models"


class LiveWrap(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, frame_u8):                       # (H, W, 3) uint8
        x = frame_u8.permute(2, 0, 1).unsqueeze(0).float() / 255.0
        y = self.net(x).clamp(0, 1)[0].permute(1, 2, 0)
        return (y * 255.0).round().to(torch.uint8)     # (2H, 2W, 3) uint8


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (ck, nf, nc) in CKPTS.items():
        p = ROOT / "models" / ck
        if not p.exists():
            print(f"skip {name}: {p} not found"); continue
        d = torch.load(p, map_location="cpu", weights_only=False)
        net = build_generator("srvgg", nf=nf, nconv=nc, scale=2).eval()
        net.load_state_dict(d["ema"])
        w = LiveWrap(net).eval()
        dummy = torch.randint(0, 255, (540, 960, 3), dtype=torch.uint8)
        out = OUT / f"pixelforge_{name}_x2.onnx"
        torch.onnx.export(w, dummy, str(out), opset_version=17, input_names=["frame"], output_names=["out"],
                          dynamic_axes={"frame": {0: "h", 1: "w"}, "out": {0: "H", 1: "W"}})
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB) from {ck}")


if __name__ == "__main__":
    main()
