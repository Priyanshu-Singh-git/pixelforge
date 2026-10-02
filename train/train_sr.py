"""Train a x4 game-texture super-resolution generator.

Stage "psnr": L1 loss only (sharp-but-safe, maximises fidelity).
Stage "gan" : init from a psnr checkpoint, then L1 + VGG19 perceptual + adversarial loss with the
              U-Net discriminator (Real-ESRGAN recipe, scaled down). Generator EMA is evaluated/saved.

  python train/train_sr.py --name rrdb_s_psnr --arch rrdb --nf 32 --nb 6 --stage psnr --iters 20000
  python train/train_sr.py --name rrdb_s_gan  --arch rrdb --nf 32 --nb 6 --stage gan --init models/rrdb_s_psnr.pt --iters 10000
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.data import TrainPatches, build_eval_set  # noqa: E402
from pixelforge.metrics import lpips_dist, psnr  # noqa: E402
from pixelforge.models import UNetDiscriminatorSN, build_generator, n_params  # noqa: E402


class VGGPerceptual(torch.nn.Module):
    """L1 between pre-ReLU VGG19 features (conv1_2, conv2_2, conv3_4, conv4_4, conv5_4)."""
    LAYERS = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}

    def __init__(self):
        super().__init__()
        from torchvision.models import VGG19_Weights, vgg19
        self.vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:35].eval().requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        x, y = (x - self.mean) / self.std, (y - self.mean) / self.std
        loss = 0.0
        for i, layer in enumerate(self.vgg):
            x, y = layer(x), layer(y)
            if i in self.LAYERS:
                loss = loss + self.LAYERS[i] * F.l1_loss(x, y.detach())
        return loss


@torch.no_grad()
def validate(net, val, device, with_lpips=False, n=None):
    net.eval()
    ps, lp = [], []
    for k in range(len(val["lr"]) if n is None else n):
        lr = torch.from_numpy(val["lr"][k]).permute(2, 0, 1)[None].float().div(255).to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            sr = net(lr)
        sr = (sr.float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        ps.append(psnr(sr, val["hr"][k]))
        if with_lpips:
            lp.append(lpips_dist(sr, val["hr"][k], device))
    net.train()
    return float(np.mean(ps)), (float(np.mean(lp)) if lp else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="rrdb", choices=["rrdb", "srvgg"])
    ap.add_argument("--nf", type=int, default=32)
    ap.add_argument("--nb", type=int, default=6)
    ap.add_argument("--gc", type=int, default=16)
    ap.add_argument("--nconv", type=int, default=16)
    ap.add_argument("--stage", default="psnr", choices=["psnr", "gan"])
    ap.add_argument("--init", default=None)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--hr", type=int, default=128)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--degrade", default="real", choices=["real", "bicubic", "frame"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = dict(arch=args.arch, nf=args.nf, nb=args.nb, gc=args.gc, nconv=args.nconv)
    if args.scale != 4:
        cfg["scale"] = args.scale
    G = build_generator(**cfg).to(device)
    if args.init:
        G.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["ema"])
    G_ema = copy.deepcopy(G).eval().requires_grad_(False)
    gan = args.stage == "gan"
    lr = args.lr or (1e-4 if gan else 2e-4)
    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda i: 0.5 * (1 + math.cos(math.pi * min(i / args.iters, 1))))
    scaler = torch.cuda.amp.GradScaler()
    if gan:
        D = UNetDiscriminatorSN().to(device)
        opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.9, 0.99))
        percep = VGGPerceptual().to(device)
    ema_decay = 0.999

    loader = DataLoader(TrainPatches(hr_size=args.hr, scale=args.scale, mode=args.degrade, length=args.iters * args.bs, seed=args.seed),
                        batch_size=args.bs, num_workers=args.workers, pin_memory=True, drop_last=True,
                        persistent_workers=True)
    val = build_eval_set("val", args.degrade, scale=args.scale)
    log = ROOT / "train" / "runs" / f"{args.name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    out = ROOT / "models" / f"{args.name}.pt"
    best = -1e9
    print(f"[{args.name}] {args.stage} {cfg} params={n_params(G)/1e6:.2f}M iters={args.iters} bs={args.bs}", flush=True)
    t0 = time.perf_counter()
    for it, (lr_img, hr_img) in enumerate(loader, 1):
        lr_img, hr_img = lr_img.to(device, non_blocking=True), hr_img.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            sr = G(lr_img)
            l_pix = F.l1_loss(sr.float(), hr_img)
            loss = l_pix
            if gan:
                l_per = percep(sr.float(), hr_img)
                d_out = D(sr).float()
                l_adv = F.binary_cross_entropy_with_logits(d_out, torch.ones_like(d_out))
                loss = l_pix + l_per + 0.1 * l_adv
        opt_g.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt_g)
        if gan:  # discriminator step: real -> 1, fake -> 0
            with torch.autocast("cuda", dtype=torch.float16):
                d_real = D(hr_img).float()
                d_fake = D(sr.detach()).float()
                l_d = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                       + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
            opt_d.zero_grad(set_to_none=True)
            scaler.scale(l_d).backward()
            scaler.step(opt_d)
        scaler.update()
        sched.step()
        with torch.no_grad():
            decay = min(ema_decay, (1 + it) / (10 + it))  # EMA warm-up
            for pe, p in zip(G_ema.parameters(), G.parameters()):
                pe.lerp_(p, 1 - decay)
            for be, b in zip(G_ema.buffers(), G.buffers()):
                be.copy_(b)
        if it % 200 == 0:
            el = time.perf_counter() - t0
            print(f"[{args.name}] it {it}/{args.iters} l1={l_pix.item():.4f}"
                  + (f" per={l_per.item():.3f} adv={l_adv.item():.3f} d={l_d.item():.3f}" if gan else "")
                  + f" {it / el:.1f} it/s eta {(args.iters - it) / (it / el) / 60:.0f} min", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            v_psnr, v_lpips = validate(G_ema, val, device, with_lpips=gan)
            score = -v_lpips if gan else v_psnr  # select on perceptual quality for GAN, fidelity otherwise
            row = {"it": it, "val_psnr": v_psnr, "val_lpips": v_lpips, "minutes": (time.perf_counter() - t0) / 60}
            with open(log, "a") as f:
                f.write(json.dumps(row) + "\n")
            if score > best:
                best = score
                torch.save({"ema": G_ema.state_dict(), "config": cfg, "stage": args.stage, "it": it,
                            "val": row, "params": n_params(G)}, out)
            print(f"[{args.name}] VAL it {it}: psnr={v_psnr:.3f}" + (f" lpips={v_lpips:.4f}" if gan else "")
                  + (" (best, saved)" if score == best else ""), flush=True)
    print(f"[{args.name}] done in {(time.perf_counter() - t0) / 60:.1f} min -> {out}", flush=True)


if __name__ == "__main__":
    main()
