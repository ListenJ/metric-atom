"""
Phase 4 网格搜索: w_direct × sinkhorn_eps
===========================================
固定参数: w_met=0.005, w_vol=0.1, ent_weight=0.005, num_epochs=200
扫描范围:
  w_direct    ∈ [0.5, 1.0, 2.0, 5.0]
  sinkhorn_eps ∈ [0.05, 0.1, 0.2, 0.35, 0.5]
总计: 4 × 5 = 20 runs

每个 run 用 8 个 seed (100-107) 评估方差。
输出: outputs/phase4/results.csv + 每个 run 的日志
"""
import sys, os, csv, time, gc, itertools, json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from train_2d import train_scene

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()

# ── 网格定义 ──
W_DIRECT_VALUES    = [0.5, 1.0, 2.0, 5.0]
SINKHORN_EPS_VALUES = [0.05, 0.1, 0.2, 0.35, 0.5]
SEEDS = list(range(100, 108))  # 8 seeds

# ── 固定参数 ──
FIXED = dict(
    H=64, W=64,
    num_atoms=111,
    num_epochs=200,
    num_views=8,
    num_objects=2,
    phase2_start=50,        # ECO 初始化提前
    lr=1e-3,
    bf16=BF16,
    num_samples=128,
    seed_every=25,
    quick_mode=True,
    # 固定超参
    w_met=0.005,
    w_vol=0.1,
    w_coh=2.0,
    w_pos=5.0,
    ent_weight=0.005,
    use_direct_loss=True,
    use_eco=False,           # Phase 4 只用 DirectCluster，不加 ECO
    diff_K=5,
)

RESULTS_DIR = Path('outputs/phase4')
RESULTS_CSV = RESULTS_DIR / 'results.csv'
RESULTS_JSON = RESULTS_DIR / 'summary.json'


def run_single(w_direct, sinkhorn_eps, seed):
    """运行单次实验，返回 metrics dict"""
    output_dir = RESULTS_DIR / f'w{w_direct}_eps{sinkhorn_eps}_seed{seed}'
    params = {
        **FIXED,
        'w_direct': w_direct,
        'sinkhorn_eps': sinkhorn_eps,
        'seed': seed,
        'output_dir': str(output_dir),
        'device': DEVICE,
    }

    start = time.time()
    try:
        atoms, field, log, metrics = train_scene(**params)
        elapsed = time.time() - start
        return {
            'success': True,
            'ARI': metrics.get('ARI', float('nan')),
            'NMI': metrics.get('NMI', float('nan')),
            'valid_atoms': metrics.get('valid_atoms', 0),
            'total_atoms': metrics.get('total_atoms', 0),
            'elapsed_min': round(elapsed / 60, 1),
        }
    except Exception as e:
        elapsed = time.time() - start
        print(f"    FAILED: {e}")
        return {
            'success': False,
            'ARI': float('nan'),
            'NMI': float('nan'),
            'valid_atoms': 0,
            'total_atoms': 0,
            'elapsed_min': round(elapsed / 60, 1),
            'error': str(e),
        }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(W_DIRECT_VALUES, SINKHORN_EPS_VALUES))
    total_runs = len(grid) * len(SEEDS)
    print(f"Phase 4 网格搜索")
    print(f"  w_direct:     {W_DIRECT_VALUES}")
    print(f"  sinkhorn_eps: {SINKHORN_EPS_VALUES}")
    print(f"  seeds:        {SEEDS}")
    print(f"  总 runs:      {total_runs}")
    print(f"  设备:         {DEVICE} (BF16={BF16})")
    print()

    all_results = []
    run_idx = 0

    for w_direct, sinkhorn_eps in grid:
        config_results = []
        print(f"\n{'='*60}")
        print(f"  w_direct={w_direct}, sinkhorn_eps={sinkhorn_eps}")
        print(f"{'='*60}")

        for seed in SEEDS:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] seed={seed} ... ", end='', flush=True)

            result = run_single(w_direct, sinkhorn_eps, seed)
            config_results.append({
                'w_direct': w_direct,
                'sinkhorn_eps': sinkhorn_eps,
                'seed': seed,
                **result,
            })

            if result['success']:
                print(f"ARI={result['ARI']:.4f}  NMI={result['NMI']:.4f}  valid={result['valid_atoms']}/{result['total_atoms']}  ({result['elapsed_min']} min)")
            else:
                print(f"FAILED ({result.get('error', 'unknown')})")

            all_results.append(config_results[-1])

            # 增量保存 CSV
            with open(RESULTS_CSV, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'w_direct', 'sinkhorn_eps', 'seed',
                    'success', 'ARI', 'NMI', 'valid_atoms', 'total_atoms', 'elapsed_min', 'error'
                ])
                writer.writeheader()
                writer.writerows(all_results)

            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()

        # 当前配置汇总
        aris = [r['ARI'] for r in config_results if r['success']]
        if aris:
            import numpy as np
            mean_ari = np.mean(aris)
            std_ari = np.std(aris)
            median_ari = np.median(aris)
            n_good = sum(1 for a in aris if a >= 0.5)
            print(f"\n  >>> 汇总: mean_ARI={mean_ari:.4f}  std={std_ari:.4f}  median={median_ari:.4f}  >=0.5: {n_good}/{len(aris)}")

    # ── 最终汇总 ──
    print(f"\n{'='*60}")
    print("PHASE 4 COMPLETE — 配置汇总")
    print(f"{'='*60}")

    summary = {}
    for w_direct, sinkhorn_eps in grid:
        config_results = [r for r in all_results
                         if r['w_direct'] == w_direct and r['sinkhorn_eps'] == sinkhorn_eps and r['success']]
        aris = [r['ARI'] for r in config_results]
        if aris:
            import numpy as np
            summary[f"w{w_direct}_eps{sinkhorn_eps}"] = {
                'w_direct': w_direct,
                'sinkhorn_eps': sinkhorn_eps,
                'mean_ARI': float(np.mean(aris)),
                'std_ARI': float(np.std(aris)),
                'median_ARI': float(np.median(aris)),
                'max_ARI': float(np.max(aris)),
                'min_ARI': float(np.min(aris)),
                'n_runs': len(aris),
                'n_good': sum(1 for a in aris if a >= 0.5),
            }

    # 按 mean_ARI 排序打印
    sorted_configs = sorted(summary.values(), key=lambda x: x['mean_ARI'], reverse=True)
    print(f"\n{'配置':<25} {'mean_ARI':>10} {'std':>8} {'median':>8} {'max':>8} {'>=0.5':>6}")
    print("-" * 70)
    for cfg in sorted_configs:
        label = f"w{cfg['w_direct']}_eps{cfg['sinkhorn_eps']}"
        print(f"{label:<25} {cfg['mean_ARI']:>10.4f} {cfg['std_ARI']:>8.4f} "
              f"{cfg['median_ARI']:>8.4f} {cfg['max_ARI']:>8.4f} {cfg['n_good']}/{cfg['n_runs']}")

    # 保存 JSON 汇总
    with open(RESULTS_JSON, 'w') as f:
        json.dump({
            'phase': 4,
            'grid': {'w_direct': W_DIRECT_VALUES, 'sinkhorn_eps': SINKHORN_EPS_VALUES},
            'seeds': SEEDS,
            'configs': summary,
            'best_config': sorted_configs[0] if sorted_configs else None,
        }, f, indent=2)

    print(f"\n结果已保存:")
    print(f"  CSV:  {RESULTS_CSV}")
    print(f"  JSON: {RESULTS_JSON}")


if __name__ == '__main__':
    main()
