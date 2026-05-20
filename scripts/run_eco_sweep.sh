#!/bin/bash
# ECO weight sweep: test w_eco ∈ {0.1, 0.3, 0.5, 1.0}
# Each weight runs 8 seeds × 200 epochs on 64×64

cd /root/MetricAtom
BASE_DIR=outputs/eco_sweep
mkdir -p ${BASE_DIR}

for W_ECO in 0.1 0.3 0.5 1.0; do
    echo "================================================================"
    echo "W_ECO=${W_ECO} START: $(date)"
    echo "================================================================"

    for seed in 100 101 102 103 104 105 106 107; do
        echo "  seed ${seed} start: $(date)"
        rm -rf ${BASE_DIR}/w${W_ECO}_seed${seed}
        mkdir -p ${BASE_DIR}/w${W_ECO}_seed${seed}

        /root/miniconda3/bin/python -u train_2d.py \
            --resolution 64 --epochs 200 --atom 100 \
            --bf16 --seed ${seed} --samples 64 \
            --use-eco --eco-id-weight 0.1 --w-eco ${W_ECO} \
            --quick \
            --output ${BASE_DIR}/w${W_ECO}_seed${seed} \
            > ${BASE_DIR}/w${W_ECO}_seed${seed}.log 2>&1

        ARI=$(grep "ARI:" ${BASE_DIR}/w${W_ECO}_seed${seed}.log | head -1)
        echo "  seed ${seed} result: $ARI"
    done

    echo ""
    echo "=== W_ECO=${W_ECO} SUMMARY ==="
    for seed in 100 101 102 103 104 105 106 107; do
        ARI=$(grep "ARI:" ${BASE_DIR}/w${W_ECO}_seed${seed}.log | head -1)
        echo "  ${seed}: ${ARI}"
    done
    echo ""
done

echo "================================================================"
echo "ALL DONE: $(date)"
echo "================================================================"

# Final summary
echo ""
echo "=== ECO WEIGHT SWEEP RESULTS ==="
for W_ECO in 0.1 0.3 0.5 1.0; do
    echo "--- w_eco=${W_ECO} ---"
    for seed in 100 101 102 103 104 105 106 107; do
        ARI=$(grep "ARI:" ${BASE_DIR}/w${W_ECO}_seed${seed}.log | head -1 | awk '{print $2}')
        echo "  ${seed}: ${ARI}"
    done
done
