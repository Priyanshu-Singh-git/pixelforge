"""Synthetic degradations that turn a clean HR texture into a realistic low-res one.

"real" mode is a simplified Real-ESRGAN first-order pipeline (Wang et al., 2021):
  blur -> random-kernel downscale (x4) -> noise -> JPEG
"bicubic" mode is the classic clean SR setting (bicubic x4 only).
"""
from __future__ import annotations

import cv2
import numpy as np

INTERP = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
INTER_FRAME = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_NEAREST]


def degrade(hr: np.ndarray, rng: np.random.Generator, scale: int = 4, mode: str = "real") -> np.ndarray:
    """hr: HxWx3 uint8 RGB (H, W divisible by scale) -> (H/scale)x(W/scale)x3 uint8."""
    h, w = hr.shape[:2]
    size = (w // scale, h // scale)
    if mode == "bicubic":
        return cv2.resize(hr, size, interpolation=cv2.INTER_CUBIC)
    if mode == "frame":  # a game rendered at lower resolution: clean but soft/aliased, no noise or JPEG
        img = hr.astype(np.float32)
        if rng.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), sigmaX=rng.uniform(0.2, 0.8))
        img = cv2.resize(img, size, interpolation=INTER_FRAME[rng.integers(len(INTER_FRAME))])
        return np.clip(img, 0, 255).round().astype(np.uint8)
    img = hr.astype(np.float32)
    if rng.random() < 0.8:  # isotropic / anisotropic Gaussian blur
        sx = rng.uniform(0.2, 2.0)
        sy = sx if rng.random() < 0.6 else rng.uniform(0.2, 2.0)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=sx, sigmaY=sy)
    img = cv2.resize(img, size, interpolation=INTERP[rng.integers(len(INTERP))])
    if rng.random() < 0.5:  # Gaussian noise, colour or grey
        sigma = rng.uniform(1, 10)
        noise = rng.normal(0, sigma, img.shape if rng.random() < 0.6 else img.shape[:2] + (1,))
        img = img + noise
    img = np.clip(img, 0, 255).round().astype(np.uint8)
    if rng.random() < 0.7:  # JPEG compression (game assets are often stored lossy)
        q = int(rng.integers(40, 96))
        ok, buf = cv2.imencode(".jpg", img[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img)
