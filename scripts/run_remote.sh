#!/bin/bash
set -e
cd /root/MetricAtom
PYTHON=/root/miniconda3/bin/python
OUTDIR=outputs/exp_remote
mkdir -p $OUTDIR

BASE="--resolution 64 --epochs 500"
HPS="--fp16 --w-met 0.005 --w-vol 0.1 --w-direct 2.0 --sinkhorn-eps 0.5 --sinkhorn-iters 50 --diff-k 5"

cleanup() {
  $PYTHON -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); print('VRAM freed')"
}

echo "===== [1/4] seed 123 ====="
$PYTHON train_2d.py $BASE --quick $HPS --seed 123 --output $OUTDIR/seed123 2>&1 | tee $OUTDIR/seed123.log
cleanup
sleep 2

echo "===== [2/4] seed 456 ====="
$PYTHON train_2d.py $BASE --quick $HPS --seed 456 --output $OUTDIR/seed456 2>&1 | tee $OUTDIR/seed456.log
cleanup
sleep 2

echo "===== [3/4] 3-object 64x64 ====="
$PYTHON train_2d.py $BASE --quick $HPS --num-objects 3 --seed 42 --output $OUTDIR/3obj 2>&1 | tee $OUTDIR/3obj.log
cleanup
sleep 2

echo "===== [4/4] 128x128 (T4 16GB, chunked) ====="
$PYTHON train_2d.py --resolution 128 --epochs 800 --num-samples 64 \
  $HPS --seed 42 --chunk-size 2048 --output $OUTDIR/128x128 2>&1 | \
  tee $OUTDIR/128x128.log
cleanup

echo "===== ALL DONE ====="
grep -E "Clustering|ARI:|NMI:|Valid atoms" $OUTDIR/*.log 2>/dev/null
