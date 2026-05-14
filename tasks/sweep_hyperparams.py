"""
MetricAtom 超参网格扫描脚本。

对对比学习 + 损失权重超参空间做随机搜索，
记录每次训练的 ARI 到 CSV，用于找最优组合。

用法:
    python tasks/sweep_hyperparams.py --n-trials 20
    python tasks/sweep_hyperparams.py --dry-run          # 只打印组合
    python tasks/sweep_hyperparams.py --resume results.csv   # 续跑
"""

import sys, os, random, csv, math, time, json
from pathlib import Path
from datetime import datetime

# ── 让 Python 能找到项目根目录 ──
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from train_2d import train_scene


# ── 搜索空间 ──
SEARCH_SPACE = {
    'tau':         [0.1, 0.2, 0.3, 0.5],
    'pos_thresh':  [0.2, 0.3, 0.5],
    'neg_thresh':  [1.0, 1.5, 2.0, 3.0],
    'var_weight':  [0.05, 0.1, 0.2, 0.5],
    'w_coh':       [0.5, 1.0, 2.0, 5.0],
    'w_vol':       [0.1, 0.2, 0.5],
}

CSV_HEADER = [
    'run_id', 'tau', 'pos_thresh', 'neg_thresh', 'var_weight',
    'w_coh', 'w_vol', 'ARI', 'NMI', 'valid_atoms', 'total_atoms',
    'elapsed_min', 'epochs_completed', 'error'
]

# 固定参数（快速扫描用）
H, W = 64, 64
N_ATOMS = 100
N_EPOCHS = 200          # 比全量 600 少，加快扫描
PHASE2_START = 80        # ~40%
LR = 1e-3
NUM_SAMPLES = 64
NUM_VIEWS = 8


def generate_random_trials(n, space):
    """从搜索空间中采样 n 个不重复的随机组合。"""
    keys = list(space.keys())
    trials = set()
    results = []
    max_possible = math.prod(len(space[k]) for k in keys)
    n = min(n, max_possible)
    
    while len(results) < n:
        trial = tuple(random.choice(space[k]) for k in keys)
        if trial not in trials:
            trials.add(trial)
            results.append(dict(zip(keys, trial)))
    return results


def load_existing_results(csv_path):
    """加载已有 CSV，返回 {tuple(params): row_dict}，用于断点续跑。"""
    if not os.path.exists(csv_path):
        return {}
    completed = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (float(row['tau']), float(row['pos_thresh']),
                   float(row['neg_thresh']), float(row['var_weight']),
                   float(row['w_coh']), float(row['w_vol']))
            completed[key] = row
    print(f"  已加载 {len(completed)} 条已有结果 (跳过已完成组合)")
    return completed


def run_trial(params, run_id, output_root):
    """运行一次训练，返回指标 dict。"""
    output_dir = output_root / f"run_{run_id:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    t0 = time.time()
    error = ""
    ari, nmi = float('nan'), float('nan')
    valid_atoms, total_atoms = 0, 0
    
    try:
        _, _, _, metrics = train_scene(
            H=H, W=W,
            num_atoms=N_ATOMS,
            num_epochs=N_EPOCHS,
            num_views=NUM_VIEWS,
            phase2_start=PHASE2_START,
            lr=LR,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            output_dir=str(output_dir),
            bf16=True,
            num_samples=NUM_SAMPLES,
            seed_every=25,
            quick_mode=True,
            **params
        )
        ari = metrics.get('ARI', float('nan'))
        nmi = metrics.get('NMI', float('nan'))
        valid_atoms = metrics.get('valid_atoms', 0)
        total_atoms = metrics.get('total_atoms', 0)
        
    except RuntimeError as e:
        err_str = str(e)
        if 'out of memory' in err_str.lower() or 'CUDA' in err_str:
            error = "CUDA_OOM"
            torch.cuda.empty_cache()
        else:
            error = f"RuntimeError: {err_str[:200]}"
    except Exception as e:
        error = f"{type(e).__name__}: {str(e)[:200]}"
    
    elapsed = (time.time() - t0) / 60.0
    
    return {
        'run_id': run_id,
        **params,
        'ARI': ari,
        'NMI': nmi,
        'valid_atoms': valid_atoms,
        'total_atoms': total_atoms,
        'elapsed_min': round(elapsed, 2),
        'epochs_completed': N_EPOCHS,
        'error': error,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='MetricAtom 超参随机搜索')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='随机搜索次数 (默认 20)')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印组合，不运行')
    parser.add_argument('--resume', type=str, default=None,
                        help='已有 CSV 路径，跳过已完成的组合')
    args = parser.parse_args()
    
    # ── 输出目录 ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_root = Path('outputs') / f'sweep_{timestamp}'
    output_root.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_root / 'results.csv'
    
    # ── 生成 / 加载 trials ──
    existing = {}
    if args.resume:
        existing = load_existing_results(args.resume)
        # 使用已有 CSV 的目录作为输出
        output_root = Path(args.resume).parent
        csv_path = Path(args.resume)
    
    trials = generate_random_trials(args.n_trials, SEARCH_SPACE)
    
    # ── 过滤已完成的 ──
    filtered = []
    for t in trials:
        key = (t['tau'], t['pos_thresh'], t['neg_thresh'],
               t['var_weight'], t['w_coh'], t['w_vol'])
        if key not in existing:
            filtered.append(t)
    
    if len(filtered) == 0:
        print("所有组合已完成！")
        return
    
    # ── 保存搜索空间元信息 ──
    with open(output_root / 'search_space.json', 'w') as f:
        json.dump({'space': {k: v for k, v in SEARCH_SPACE.items()},
                    'n_trials': args.n_trials,
                    'fixed_params': {'H': H, 'W': W, 'epochs': N_EPOCHS,
                                     'phase2_start': PHASE2_START, 'lr': LR,
                                     'num_atoms': N_ATOMS, 'num_samples': NUM_SAMPLES}},
                  f, indent=2)
    
    if args.dry_run:
        print(f"\n[Dry Run] 计划运行 {len(filtered)} 个组合 (共 {args.n_trials}):")
        for t in filtered:
            print(f"  tau={t['tau']:.1f} pt={t['pos_thresh']:.1f} nt={t['neg_thresh']:.1f} "
                  f"vw={t['var_weight']:.2f} wc={t['w_coh']:.1f} wv={t['w_vol']:.1f}")
        print(f"\n输出目录: {output_root}")
        return
    
    print(f"\n{'='*70}")
    print(f"MetricAtom 超参扫描启动")
    print(f"  组合: {len(filtered)}/{args.n_trials} (已跳过 {len(trials) - len(filtered)} 个已完成)")
    print(f"  训练: {H}x{W}, {N_EPOCHS} epochs, Phase2 @ {PHASE2_START}")
    print(f"  输出: {output_root}")
    print(f"{'='*70}\n")
    
    # ── 写入 CSV header (如果文件不存在) ──
    if not csv_path.exists():
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
    
    # ── 运行 ──
    best_ari = -float('inf')
    best_row = None
    
    for i, params in enumerate(filtered):
        run_id = i + 1 + len(existing)
        print(f"\n[Run {run_id}/{len(filtered) + len(existing)}] "
              f"tau={params['tau']:.1f} pt={params['pos_thresh']:.1f} "
              f"nt={params['neg_thresh']:.1f} vw={params['var_weight']:.2f} "
              f"wc={params['w_coh']:.1f} wv={params['w_vol']:.1f}")
        
        row = run_trial(params, run_id, output_root)
        
        # 追加到 CSV
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([row[h] for h in CSV_HEADER])
        
        ari = row['ARI']
        err = row['error']
        
        if not math.isnan(ari) and ari > best_ari:
            best_ari = ari
            best_row = row
        
        status = f"ARI={ari:.4f}" if not math.isnan(ari) else f"ARI=NaN"
        if err:
            status += f" [ERR: {err}]"
        status += f" ({row['elapsed_min']:.1f}m)"
        print(f"  -> {status}")
    
    # ── 总结 ──
    print(f"\n{'='*70}")
    print(f"扫描完成！结果: {csv_path}")
    if best_row and not math.isnan(best_ari):
        print(f"最佳组合 (ARI={best_ari:.4f}):")
        for k in ['tau', 'pos_thresh', 'neg_thresh', 'var_weight', 'w_coh', 'w_vol']:
            print(f"  {k}: {best_row[k]}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
