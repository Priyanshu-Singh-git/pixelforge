"""Training patches (degraded on the fly) and fixed evaluation sets."""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degrade import degrade

ROOT = Path(__file__).resolve().parents[2]


def load_manifest(split: str) -> list[dict]:
    return [m for m in json.loads((ROOT / "data" / "manifest.json").read_text()) if m["split"] == split]


def read_rgb(path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return np.ascontiguousarray(img[..., ::-1])


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).float() / 255.0


class TrainPatches(Dataset):
    """Random HR crops with flips/rotations; LR made by `degrade` in the worker."""

    def __init__(self, hr_size=128, scale=4, length=100_000, mode="real", seed=0):
        self.items = [ROOT / "data" / "polyhaven" / f"{m['slug']}.jpg" for m in load_manifest("train")]
        self.hr, self.scale, self.length, self.mode, self.seed = hr_size, scale, length, mode, seed

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed, i))
        img = read_rgb(self.items[rng.integers(len(self.items))])
        # random rescale of the source so patches cover several texel densities
        s = rng.uniform(0.5, 1.0)
        if s < 0.98:
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y, x = rng.integers(0, h - self.hr + 1), rng.integers(0, w - self.hr + 1)
        hr = img[y:y + self.hr, x:x + self.hr]
        if rng.random() < 0.5:
            hr = hr[:, ::-1]
        hr = np.ascontiguousarray(np.rot90(hr, rng.integers(4)))
        lr = degrade(hr, rng, self.scale, self.mode)
        return to_tensor(lr), to_tensor(hr)


def build_eval_set(split: str, mode: str, hr_size=512, scale=4) -> dict:
    """Centre HR crop per texture + a degradation seeded by the texture name (fixed forever).
    Cached to data/eval_<split>_<mode>.npz so every model sees identical inputs."""
    cache = ROOT / "data" / f"eval_{split}_{mode}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    lrs, hrs, slugs = [], [], []
    for m in load_manifest(split):
        img = read_rgb(ROOT / "data" / "polyhaven" / f"{m['slug']}.jpg")
        h, w = img.shape[:2]
        y, x = (h - hr_size) // 2, (w - hr_size) // 2
        hr = np.ascontiguousarray(img[y:y + hr_size, x:x + hr_size])
        rng = np.random.default_rng(zlib.crc32(m["slug"].encode()))
        lrs.append(degrade(hr, rng, scale, mode)); hrs.append(hr); slugs.append(m["slug"])
    d = {"lr": np.stack(lrs), "hr": np.stack(hrs), "slug": np.array(slugs)}
    np.savez(cache, **d)
    return d
