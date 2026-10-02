import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.degrade import degrade  # noqa: E402
from pixelforge.metrics import psnr, ssim  # noqa: E402
from pixelforge.models import RRDBNet, SRVGGNetCompact, UNetDiscriminatorSN, build_generator  # noqa: E402


@pytest.mark.parametrize("mode", ["real", "bicubic"])
def test_degrade_shape_dtype_and_determinism(mode):
    hr = (np.random.default_rng(0).random((128, 96, 3)) * 255).astype(np.uint8)
    a = degrade(hr, np.random.default_rng(1), 4, mode)
    b = degrade(hr, np.random.default_rng(1), 4, mode)
    assert a.shape == (32, 24, 3) and a.dtype == np.uint8
    assert np.array_equal(a, b)  # same seed -> same degradation (fixed eval sets rely on this)


def test_degrade_real_differs_from_clean_bicubic():
    hr = (np.random.default_rng(0).random((128, 128, 3)) * 255).astype(np.uint8)
    diffs = [np.abs(degrade(hr, np.random.default_rng(s), 4, "real").astype(int)
                    - degrade(hr, None, 4, "bicubic").astype(int)).mean() for s in range(5)]
    assert max(diffs) > 1.0


@pytest.mark.parametrize("arch,kw", [("rrdb", dict(nf=16, nb=2, gc=8)), ("srvgg", dict(nf=16, nconv=4))])
def test_generators_upscale_x4(arch, kw):
    net = build_generator(arch, **kw).eval()
    with torch.no_grad():
        y = net(torch.rand(2, 3, 20, 24))
    assert y.shape == (2, 3, 80, 96)


def test_discriminator_is_per_pixel():
    with torch.no_grad():
        y = UNetDiscriminatorSN(num_feat=16)(torch.rand(1, 3, 64, 64))
    assert y.shape == (1, 1, 64, 64)


def test_official_parameter_names():
    keys = set(RRDBNet(num_feat=8, num_block=1, num_grow_ch=4).state_dict())
    assert {"conv_first.weight", "body.0.rdb1.conv1.weight", "conv_body.weight", "conv_up1.weight",
            "conv_hr.weight", "conv_last.weight"} <= keys
    keys = set(SRVGGNetCompact(num_feat=8, num_conv=2).state_dict())
    assert {"body.0.weight", "body.1.weight", "body.6.weight"} <= keys


def test_metrics():
    a = (np.random.default_rng(0).random((64, 64, 3)) * 255).astype(np.uint8)
    assert psnr(a, a) > 90 and ssim(a, a) == pytest.approx(1.0)
    noisy = np.clip(a.astype(int) + 20, 0, 255).astype(np.uint8)
    assert psnr(noisy, a) < psnr(a, a)
