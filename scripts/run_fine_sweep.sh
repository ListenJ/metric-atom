#!/bin/bash
# Fine-tuning sweep: lower temperatures + adjusted w_eco
# Scan eps ∈ {0.02, 0.03, 0.08}, w_eco=0.5 (default)
# Also test eps=0.05 + w_eco=0.3

cd /root/MetricAtom
BASE=outputs/fine_sweep
mkdir -p ${BASE}

# Round 1: eps sweep
for EPS in 0.02 0.03 0.08; do
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

# Round 2: eps=0.05 with reduced w_eco=0.3
echo "================================================================"
echo "EPS=0.05 W-ECO=0.3 START: $(date)"
echo "================================================================"
for seed in 100 101 102 103 104 105 106 107; do
    echo "  seed ${seed} start: $(date)"
    rm -rf ${BASE}/eps005_w3_seed${seed}
    mkdir -p ${BASE}/eps005_w3_seed${seed}
    /root/miniconda3/bin/python -u train_2d.py \
        --resolution 64 --epochs 200 --atom 100 \
        --bf16 --seed ${seed} --samples 64 \
        --use-eco --eco-id-weight 0.1 --w-eco 0.3 \
        --sinkhorn-eps 0.05 \
        --quick \
        --output ${BASE}/eps005_w3_seed${seed} \
        > ${BASE}/eps005_w3_seed${seed}.log 2>&1
    ARI=$(grep "ARI:" ${BASE}/eps005_w3_seed${seed}.log | head -1)
    echo "  seed ${seed}: ${ARI}"
done

# Round 3: longer training (400 epochs) at best config
echo "================================================================"
echo "EPS=0.05 LONG TRAIN (400 epochs) START: $(date)"
echo "================================================================"
for seed in 100 105 106 107; do
    echo "  seed ${seed} start: $(date)"
    rm -rf ${BASE}/eps005_long_seed${seed}
    mkdir -p ${BASE}/eps005_long_seed${seed}
    /root/miniconda3/bin/python -u train_2d.py \
        --resolution 64 --epochs 400 --atom 100 \
        --bf16 --seed ${seed} --samples 64 \
        --use-eco --eco-id-weight 0.1 --w-eco 0.5 \
        --sinkhorn-eps 0.05 \
        --quick \
        --output ${BASE}/eps005_long_seed${seed} \
        > ${BASE}/eps005_long_seed${seed}.log 2>&1
    ARI=$(grep "ARI:" ${BASE}/eps005_long_seed${seed}.log | head -1)
    echo "  seed ${seed}: ${ARI}"
done

echo "================================================================"
echo "ALL DONE: $(date)"
echo "================================================================"
echo ""
echo "=== FINAL RESULTS ==="
echo "--- eps sweep ---"
for EPS in 0.02 0.03 0.08; do
    echo "eps=${EPS}:"
    for seed in 100 101 102 103 104 105 106 107; do
        ARI=$(grep "ARI:" ${BASE}/eps${EPS}_seed${seed}.log | head -1 | awk '{print $2}')
        echo "  ${seed}: ${ARI}"
    done
done
echo "--- eps=0.05 w_eco=0.3 ---"
for seed in 100 101 102 103 104 105 106 107; do
    ARI=$(grep "ARI:" ${BASE}/eps005_w3_seed${seed}.log | head -1 | awk '{print $2}')
    echo "  ${seed}: ${ARI}"
done
echo "--- eps=0.05 long train ---"
for seed in 100 105 106 107; do
    ARI=$(grep "ARI:" ${BASE}/eps005_long_seed${seed}.log | head -1 | awk '{print $2}')
    echo "  ${seed}: ${ARI}"
done
