#!/bin/bash
# Temperature (sinkhorn_eps) sweep on z-score ECO + balanced KMeans baseline
cd /root/MetricAtom
BASE=outputs/temp_sweep
mkdir -p ${BASE}

for EPS in 0.05 0.1 0.2 0.5; do
    echo "================================================================"
    echo "EPS=${EPS} START: $(date)"
    echo "================================================================"

    for seed in 100 101 102 103 104 105 106 107; do
        echo "  seed ${seed} start: $(date)"
        rm -rf ${BASE}/eps${EPS}_seed${seed}
        mkdir -p ${BASE}/eps${EPS}_seed${seed}

        /root/miniconda3/bin/python -u train_2d.py \
            --resolution 64 --epochs 200 --atom 100 \
            --bf16 --seed ${seed} --samples 64 \
            --use-eco --eco-id-weight 0.1 --w-eco 0.5 \
            --sinkhorn-eps ${EPS} \
            --quick \
            --output ${BASE}/eps${EPS}_seed${seed} \
            > ${BASE}/eps${EPS}_seed${seed}.log 2>&1

        ARI=$(grep "ARI:" ${BASE}/eps${EPS}_seed${seed}.log | head -1)
        echo "  seed ${seed}: ${ARI}"
    done

    echo ""
    echo "=== EPS=${EPS} SUMMARY ==="
    for seed in 100 101 102 103 104 105 106 107; do
        ARI=$(grep "ARI:" ${BASE}/eps${EPS}_seed${seed}.log | head -1)
        echo "  ${seed}: ${ARI}"
    done
done

echo "================================================================"
echo "=== FINAL RESULTS ==="
for EPS in 0.05 0.1 0.2 0.5; do
    echo "--- eps=${EPS} ---"
    for seed in 100 101 102 103 104 105 106 107; do
        ARI=$(grep "ARI:" ${BASE}/eps${EPS}_seed${seed}.log | head -1 | awk '{print $2}')
        echo "  ${seed}: ${ARI}"
    done
done
