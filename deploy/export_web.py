"""Export the browser demo into docs/: ONNX generator (dynamic size), sample textures, test-set stats.

  python deploy/export_web.py --model rrdb_s_gan
Parity: ONNX Runtime vs PyTorch on every sample must agree (max abs pixel error < 1/255).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pixelforge.data import build_eval_set, load_manifest  # noqa: E402
from pixelforge.models import build_generator, n_params  # noqa: E402

OUT = ROOT / "docs"
N_SAMPLES = 6


def pick_samples(d, first=None):
    """`first` (a test slug) leads, then one test texture per category (deterministic)."""
    cats = {m["slug"]: (m["categories"] or ["other"]) for m in load_manifest("test")}
    used, picks = set(), []
    if first:
        picks.append(list(map(str, d["slug"])).index(first))
    for k, slug in enumerate(d["slug"]):
        main = next((c for c in cats[str(slug)] if c not in ("floor", "wall", "outdoor", "indoor", "man made", "natural")),
                    cats[str(slug)][0])
        if main not in used and k not in picks:
            used.add(main); picks.append(k)
        if len(picks) == N_SAMPLES:
            break
    return picks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--first", default="slumped_mortar_brick")
    args = ap.parse_args()
    for sub in ("models", "samples"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
        (OUT / sub).mkdir(parents=True)
    ck = torch.load(ROOT / "models" / f"{args.model}.pt", map_location="cpu", weights_only=False)
    net = build_generator(**ck["config"]).eval()
    net.load_state_dict(ck["ema"])
    onnx_path = OUT / "models" / f"pixelforge_{args.model}.onnx"
    torch.onnx.export(net, torch.rand(1, 3, 64, 64), str(onnx_path), opset_version=17, input_names=["lr"],
                      output_names=["sr"], dynamic_axes={"lr": {2: "h", 3: "w"}, "sr": {2: "H", 3: "W"}})
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    d = build_eval_set("test", "real")
    names = {m["slug"]: m["name"] for m in load_manifest("test")}
    samples, worst = [], 0.0
    for k in pick_samples(d, args.first):
        slug = str(d["slug"][k])
        x = torch.from_numpy(d["lr"][k]).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            ref = net(x).clamp(0, 1).numpy()
        got = np.clip(sess.run(None, {"lr": x.numpy()})[0], 0, 1)
        worst = max(worst, float(np.abs(ref - got).max()))
        cv2.imwrite(str(OUT / "samples" / f"{slug}_lr.png"), d["lr"][k][..., ::-1])
        cv2.imwrite(str(OUT / "samples" / f"{slug}_hr.jpg"), d["hr"][k][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 92])
        samples.append({"slug": slug, "name": names[slug], "lr": f"samples/{slug}_lr.png", "hr": f"samples/{slug}_hr.jpg"})
    print(f"ONNX vs PyTorch max abs error: {worst:.2e} (pixel units: {worst * 255:.3f})")
    assert worst * 255 < 1.0, "ONNX export does not match PyTorch"

    rows = [json.loads(l) for l in (ROOT / "eval" / "results.jsonl").read_text().splitlines()]
    show = ["Bicubic", "Real-ESRGAN x4plus (pretrained)", "Real-ESRGAN general-v3 (pretrained)",
            "PixelForge RRDB-S, L1 (mine)", "PixelForge RRDB-S, GAN (mine)"]
    stat_rows = []
    for name in show:
        r = next((x for x in rows if x["method"] == name and x["condition"] == "real"), None)
        if r:
            size = f"{r['params_m']:.1f}M params" if r["params_m"] else "–"
            stat_rows.append({"method": name.replace(" (mine)", " ★"), "psnr": r["psnr"], "lpips": r["lpips"],
                              "size": size, "mine": "(mine)" in name})
    meta = {"model": onnx_path.name, "params_m": n_params(net) / 1e6, "samples": samples,
            "stats": {"n": len(d["lr"]), "rows": stat_rows}}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"wrote {OUT} ({size:.1f} MB), samples: {[s['name'] for s in samples]}")


if __name__ == "__main__":
    main()
