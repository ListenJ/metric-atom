#!/bin/bash
# Feature diffusion sweep on eps=0.05 baseline
# Scan: diff_alpha ∈ {0.3, 0.7}, diff_T ∈ {4, 8}
cd /root/MetricAtom
BASE=outputs/diff_sweep
mkdir -p ${BASE}

for ALPHA in 0.3 0.7; do
  for T in 4 8; do
    TAG="a${ALPHA}_t${T}"
    echo "================================================================"
    echo "alpha=${ALPHA} T=${T} START: $(date)"
    echo "================================================================"

    for seed in 100 101 102 103 104 105 106 107; do
      echo "  seed ${seed} start: $(date)"
      rm -rf ${BASE}/${TAG}_seed${seed}
      mkdir -p ${BASE}/${TAG}_seed${seed}
      /root/miniconda3/bin/python -u train_2d.py \
        --resolution 64 --epochs 200 --atom 100 \
        --bf16 --seed ${seed} --samples 64 \
        --use-eco --eco-id-weight 0.1 --w-eco 0.5 \
        --sinkhorn-eps 0.05 \
        --diff-alpha ${ALPHA} --diff-t ${T} \
        --quick \
        --output ${BASE}/${TAG}_seed${seed} \
        > ${BASE}/${TAG}_seed${seed}.log 2>&1
      ARI=$(grep "ARI:" ${BASE}/${TAG}_seed${seed}.log | head -1)
      echo "  seed ${seed}: ${ARI}"
    done

    echo ""
    echo "=== alpha=${ALPHA} T=${T} SUMMARY ==="
    for seed in 100 101 102 103 104 105 106 107; do
      ARI=$(grep "ARI:" ${BASE}/${TAG}_seed${seed}.log | head -1)
      echo "  ${seed}: ${ARI}"
    done
  done
done

echo "================================================================"
echo "=== FINAL RESULTS ==="
for ALPHA in 0.3 0.7; do
  for T in 4 8; do
    echo "--- alpha=${ALPHA} T=${T} ---"
    for seed in 100 101 102 103 104 105 106 107; do
      ARI=$(grep "ARI:" ${BASE}/a${ALPHA}_t${T}_seed${seed}.log | head -1 | awk '{print $2}')
      echo "  ${seed}: ${ARI}"
    done
  done
done
