"""Super-resolution generators and the GAN discriminator.

RRDBNet        - ESRGAN generator (Wang et al., 2018), residual-in-residual dense blocks.
SRVGGNetCompact- Real-ESRGAN's compact VGG-style generator (plain convs + pixel shuffle).
UNetDiscriminatorSN - Real-ESRGAN U-Net discriminator with spectral norm (per-pixel realness).

Parameter names follow the official BasicSR / Real-ESRGAN code so published weights load
directly (used as baselines in eval/).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x  # residual scaling stabilises deep training


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4):
        super().__init__()
        assert scale == 4
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        feat = feat + self.conv_body(self.body(feat))
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body += [nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        # learn the residual on top of nearest-neighbour upsampling
        return self.upsampler(out) + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class UNetDiscriminatorSN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, skip_connection=True):
        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        a = lambda t: F.leaky_relu(t, 0.2, inplace=True)  # noqa: E731
        up = lambda t: F.interpolate(t, scale_factor=2, mode="bilinear", align_corners=False)  # noqa: E731
        x0 = a(self.conv0(x))
        x1 = a(self.conv1(x0))
        x2 = a(self.conv2(x1))
        x3 = a(self.conv3(x2))
        x4 = a(self.conv4(up(x3)))
        x4 = x4 + x2 if self.skip_connection else x4
        x5 = a(self.conv5(up(x4)))
        x5 = x5 + x1 if self.skip_connection else x5
        x6 = a(self.conv6(up(x5)))
        x6 = x6 + x0 if self.skip_connection else x6
        out = a(self.conv7(x6))
        out = a(self.conv8(out))
        return self.conv9(out)


def build_generator(arch: str, **kw) -> nn.Module:
    if arch == "rrdb":
        return RRDBNet(num_feat=kw.get("nf", 64), num_block=kw.get("nb", 23), num_grow_ch=kw.get("gc", 32))
    if arch == "srvgg":
        return SRVGGNetCompact(num_feat=kw.get("nf", 64), num_conv=kw.get("nconv", 16), upscale=kw.get("scale", 4))
    raise ValueError(arch)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
