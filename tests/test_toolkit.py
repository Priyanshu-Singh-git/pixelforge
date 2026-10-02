import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.models import build_generator  # noqa: E402
from pixelforge.toolkit import bleed_transparent, load_model, main, upscale_image, upscale_rgb  # noqa: E402

CPU = torch.device("cpu")


def tiny_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = dict(arch="rrdb", nf=8, nb=1, gc=4)
    net = build_generator(**cfg)
    p = tmp_path / "tiny.pt"
    torch.save({"ema": net.state_dict(), "config": cfg, "params": 0}, p)
    return p


def test_tiling_matches_whole_image_in_the_interior(tmp_path):
    net = load_model(tiny_ckpt(tmp_path), CPU)
    img = (np.random.default_rng(0).random((40, 52, 3)) * 255).astype(np.uint8)
    whole = upscale_rgb(net, img, CPU, tile=1000, overlap=0)
    tiled = upscale_rgb(net, img, CPU, tile=16, overlap=8)
    assert tiled.shape == whole.shape == (160, 208, 3)
    assert np.abs(tiled.astype(int) - whole.astype(int)).max() <= 2


def test_bleed_replaces_hidden_colours_only():
    rgb = np.zeros((32, 32, 3), np.uint8); rgb[:] = (255, 0, 255)           # magenta junk everywhere
    rgb[8:24, 8:24] = (40, 160, 60)                                          # green opaque square
    alpha = np.zeros((32, 32), np.uint8); alpha[8:24, 8:24] = 255
    out = bleed_transparent(rgb, alpha)
    assert np.array_equal(out[8:24, 8:24], rgb[8:24, 8:24])                  # opaque pixels untouched
    assert np.abs(out[6, 16].astype(int) - np.array([40, 160, 60])).max() < 30  # neighbour now greenish, not magenta


def test_alpha_is_kept(tmp_path):
    net = load_model(tiny_ckpt(tmp_path), CPU)
    rgba = np.zeros((20, 20, 4), np.uint8); rgba[..., :3] = 128; rgba[5:15, 5:15, 3] = 255
    out = upscale_image(net, Image.fromarray(rgba, "RGBA"), CPU)
    assert out.mode == "RGBA" and out.size == (80, 80)
    a = np.array(out)[..., 3]
    assert a[0, 0] == 0 and a[40, 40] == 255


def test_cli_keeps_folder_structure(tmp_path):
    ck = tiny_ckpt(tmp_path)
    src = tmp_path / "in"; (src / "walls").mkdir(parents=True)
    Image.fromarray(np.full((12, 12, 3), 90, np.uint8)).save(src / "walls" / "brick.jpg")
    Image.fromarray(np.full((10, 10, 3), 30, np.uint8)).save(src / "floor.png")
    rep = main(["upscale", str(src), str(tmp_path / "out"), "--model", str(ck), "--device", "cpu", "--quiet"])
    assert rep["done"] == 2 and not rep["failed"]
    assert Image.open(tmp_path / "out" / "walls" / "brick.png").size == (48, 48)
    assert Image.open(tmp_path / "out" / "floor.png").size == (40, 40)


def test_export_roundtrip(tmp_path):
    ck = tiny_ckpt(tmp_path)
    main(["export", str(tmp_path / "pf.pth"), "--model", str(ck)])
    sd = torch.load(tmp_path / "pf.pth", map_location="cpu", weights_only=False)
    assert "params_ema" in sd and "conv_first.weight" in sd["params_ema"]
    assert (tmp_path / "pf.md").exists()
    net = load_model(tmp_path / "pf.pth", CPU)
    assert net(torch.rand(1, 3, 8, 8)).shape == (1, 3, 32, 32)
