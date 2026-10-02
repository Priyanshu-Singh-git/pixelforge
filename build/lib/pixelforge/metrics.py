"""Fidelity (PSNR/SSIM) and perceptual (LPIPS) metrics on uint8 RGB images, 4-px border cropped."""
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

BORDER = 4
_lpips = None


def _crop(a):
    return a[BORDER:-BORDER, BORDER:-BORDER]


def psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    mse = np.mean((_crop(sr).astype(np.float64) - _crop(hr).astype(np.float64)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-10)))


def ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    return float(structural_similarity(_crop(sr), _crop(hr), channel_axis=2, data_range=255))


def lpips_dist(sr: np.ndarray, hr: np.ndarray, device="cuda") -> float:
    """LPIPS (AlexNet) perceptual distance; lower = looks more like the original."""
    global _lpips
    if _lpips is None:
        import lpips
        _lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    t = lambda a: torch.from_numpy(_crop(a)).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)  # noqa: E731
    with torch.no_grad():
        return float(_lpips(t(sr), t(hr)))
