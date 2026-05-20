#!/bin/bash
cd /root/MetricAtom
LOG_DIR=outputs/eco_landscape
mkdir -p ${LOG_DIR}

for seed in 100 101 102 103 104 105 106 107; do
    echo "================================================================"
    echo "SEED $seed START: $(date)"
    echo "================================================================"
    rm -rf ${LOG_DIR}/seed${seed}
    mkdir -p ${LOG_DIR}/seed${seed}
    /root/miniconda3/bin/python -u train_2d.py \
        --resolution 64 --epochs 200 --atom 100 \
        --bf16 --seed ${seed} --samples 64 \
        --use-eco --eco-id-weight 0.1 --w-eco 0.5 \
        --quick \
        --output ${LOG_DIR}/seed${seed} \
        > ${LOG_DIR}/seed${seed}.log 2>&1
    ARI=$(grep "ARI:" ${LOG_DIR}/seed${seed}.log | head -1)
    echo "SEED $seed RESULT: $ARI"
    echo ""
done

echo "================================================================"
echo "ALL DONE: $(date)"
echo "================================================================"
echo ""
echo "=== ECO+Direct LANDSCAPE SUMMARY ==="
for seed in 100 101 102 103 104 105 106 107; do
    ARI=$(grep "ARI:" ${LOG_DIR}/seed${seed}.log | head -1)
    echo "$seed: $ARI"
done
