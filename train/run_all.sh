#!/usr/bin/env bash
# Training grid used for the results. Sequential on one GPU; caches kept off the system drive.
set -e
cd "$(dirname "$0")/.."
export TORCH_HOME="$PWD/.cache/torch" TMP="$PWD/.cache/tmp" TEMP="$PWD/.cache/tmp"
mkdir -p "$TMP"
T="python -u train/train_sr.py"
[ -f models/rrdb_s_psnr.pt ]  || $T --name rrdb_s_psnr  --arch rrdb  --nf 32 --nb 6 --gc 16 --stage psnr --iters 15000
[ -f models/srvgg_s_psnr.pt ] || $T --name srvgg_s_psnr --arch srvgg --nf 32 --nconv 16    --stage psnr --iters 15000
[ -f models/rrdb_s_gan.pt ]   || $T --name rrdb_s_gan   --arch rrdb  --nf 32 --nb 6 --gc 16 --stage gan --init models/rrdb_s_psnr.pt --iters 6000 --val-every 1000
echo ALL_DONE
