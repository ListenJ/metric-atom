#!/bin/bash
set -e
cd /d/MetricAtom
PYTHON=.venv/Scripts/python.exe
OUTDIR=outputs/exp_local
mkdir -p $OUTDIR

BASE="--resolution 64 --epochs 500"
HPS="--w-met 0.005 --w-vol 0.1 --w-direct 2.0 --sinkhorn-eps 0.5 --sinkhorn-iters 50 --diff-k 5"

cleanup() {
  $PYTHON -c "import torch, gc; gc.collect(); torch.cuda.empty_cache(); print('cleaned')"
}

echo "===== [1/3] seed 123 ====="
$PYTHON train_2d.py $BASE --quick $HPS --seed 123 --output $OUTDIR/seed123 2>&1 | tee $OUTDIR/seed123.log | tail -10
cleanup

echo "===== [2/3] seed 456 ====="
$PYTHON train_2d.py $BASE --quick $HPS --seed 456 --output $OUTDIR/seed456 2>&1 | tee $OUTDIR/seed456.log | tail -10
cleanup

echo "===== [3/3] 128x128 (RTX 3050 Ti 4GB, conservative) ====="
$PYTHON train_2d.py --resolution 128 --epochs 600 --num-samples 48 \
  $HPS --seed 42 --chunk-size 1024 --output $OUTDIR/128x128 2>&1 | \
  tee $OUTDIR/128x128.log | tail -10

echo "===== ALL DONE ====="
grep -E "Clustering|ARI|NMI|Valid" $OUTDIR/*.log 2>/dev/null
