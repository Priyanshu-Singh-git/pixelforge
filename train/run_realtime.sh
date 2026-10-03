#!/usr/bin/env bash
# Real-time 2x models for live frames (game rendered at lower res): Live (GPU/WebGPU) and Lite (CPU/mobile).
set -e
cd "$(dirname "$0")/.."
export TORCH_HOME="$PWD/.cache/torch" TMP="$PWD/.cache/tmp" TEMP="$PWD/.cache/tmp"
T="python -u train/train_sr.py --arch srvgg --scale 2 --degrade frame --hr 128 --iters 8000 --val-every 1000 --stage psnr"
[ -f models/rt_live_x2.pt ] || $T --name rt_live_x2 --nf 16 --nconv 4
[ -f models/rt_lite_x2.pt ] || $T --name rt_lite_x2 --nf 8 --nconv 3
echo ALL_DONE
