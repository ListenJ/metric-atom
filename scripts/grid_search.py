"""
Grid Search Runner for MetricAtom Hyperparameter Optimization.

Usage:
    python scripts/grid_search.py --phase 1       # Phase 1: w_met × w_vol balance
    python scripts/grid_search.py --phase 1 --dry-run  # Preview configs only

Phases:
    1: w_met × w_vol balance (4×4 = 16 runs) [InfoNCE baseline]
    1.5: w_vol sweet zone scan (0.05~0.15, 5 runs)
    2: w_coh × tau sweep (16 runs)
    3: diff_K sweep (4 runs)
    4: Direct cluster loss — w_direct × sinkhorn_eps sweep (4×5 = 20 runs)
    4.5: Direct cluster loss — w_vol fine scan around best config
    5: Direct cluster loss — w_met × w_vol validation sweep
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import gc
import csv
import itertools
import time
import traceback
from pathlib import Path

from train_2d import train_scene

# ── Search spaces ──

PHASE1 = {
    'w_met': [0.005, 0.01, 0.02, 0.05],
    'w_vol': [0.1, 0.2, 0.5, 1.0],
    # fixed defaults
    'w_coh': [2.0],
    'tau': [0.5],
    'diff_K': [5],
}

PHASE1_5 = {
    # w_vol sweet zone scan around optimal 0.1
    'w_vol': [0.05, 0.075, 0.1, 0.125, 0.15],
    # fixed to best found: w_met=0.005
    'w_met': [0.005],
    'w_coh': [2.0],
    'tau': [0.5],
    'diff_K': [5],
}

PHASE2 = {
    # Fixed to best found from Phase 1 + 1.5
    'w_met': [0.005],
    'w_vol': [0.1],
    # Search space: w_coh × tau (4×4 = 16 runs)
    'w_coh': [0.5, 1.0, 2.0, 5.0],
    'tau': [0.3, 0.5, 0.7, 1.0],
    'diff_K': [5],
}

PHASE3 = {
    'diff_K': [3, 5, 10, 20],
}

# ── Direct Cluster Loss search spaces (Path 1+3) ──

PHASE4 = {
    # w_direct × sinkhorn_eps sweep
    # Fixed: w_met=0.005, w_vol=0.1 (best from Phase 1)
    'w_met': [0.005],
    'w_vol': [0.1],
    'w_direct': [0.5, 1.0, 2.0, 5.0],
    'sinkhorn_eps': [0.05, 0.1, 0.2, 0.35, 0.5],
    'ent_weight': [0.005],
    'diff_K': [5],
}

PHASE4_5 = {
    # w_vol fine scan with best direct loss config
    'w_met': [0.005],
    'w_vol': [0.05, 0.075, 0.1, 0.125, 0.15, 0.2],
    'w_direct': [2.0],
    'sinkhorn_eps': [0.1],
    'ent_weight': [0.005],
    'diff_K': [5],
}

PHASE5 = {
    # w_met × w_vol re-validation with direct loss
    'w_met': [0.002, 0.005, 0.01, 0.02],
    'w_vol': [0.05, 0.1, 0.2, 0.5],
    'w_direct': [2.0],
    'sinkhorn_eps': [0.1],
    'ent_weight': [0.005],
    'diff_K': [5],
}

BASE_DIR = Path('outputs/grid_search')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()


def build_configs(search_space):
    """Build list of config dicts from search space."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        configs.append(config)
    return configs


def config_to_tag(config):
    """Short tag for output directory."""
    parts = []
    for k in ('w_met', 'w_vol', 'w_coh', 'w_direct', 'tau', 'diff_K',
              'sinkhorn_eps', 'ent_weight'):
        if k in config:
            v = config[k]
            if isinstance(v, float):
                parts.append(f"{k}={v:.3g}".replace('.', 'p'))
            else:
                parts.append(f"{k}={v}")
    return "_".join(parts)


def run_trial(config, trial_id, phase, use_direct_loss=True):
    """Run a single trial and return metrics dict."""
    tag = config_to_tag(config)
    output_dir = BASE_DIR / f"phase{phase}" / tag

    # Base params for 64×64 quick validation
    params = {
        'H': 64, 'W': 64, 'num_atoms': 100, 'num_epochs': 500,
        'num_views': 8, 'phase2_start': 200, 'lr': 1e-3,
        'device': DEVICE, 'output_dir': str(output_dir),
        'bf16': BF16, 'num_samples': 64,
        'render_chunk_size': 4096,
        'quick_mode': True,
        # Default fixed hyperparams
        'w_pos': 5.0, 'pos_thresh': 0.3, 'neg_thresh': 2.0, 'var_weight': 0.1,
        'diff_alpha': 0.5, 'diff_T': 2,
        # Direct cluster loss (Path 1+3) — defaults
        'use_direct_loss': use_direct_loss, 'w_direct': 2.0,
        'sinkhorn_eps': 0.1, 'sinkhorn_iters': 50, 'ent_weight': 0.005,
        # InfoNCE params (fallback when use_direct_loss=False)
        'tau': 0.5, 'w_coh': 2.0,
        'seed': 42,
    }

    # Override with config from search space
    params.update(config)

    print(f"\n{'=' * 60}")
    print(f"[Trial {trial_id} | Phase {phase}] {tag}")
    for k, v in config.items():
        print(f"    {k} = {v}")
    print(f"{'=' * 60}")
    sys.stdout.flush()

    start = time.time()

    try:
        atoms, field, log, metrics = train_scene(**params)
        elapsed = time.time() - start

        result = {
            'trial_id': trial_id,
            'phase': phase,
            'tag': tag,
            'elapsed_min': f"{elapsed / 60:.1f}",
            'ARI': metrics.get('ARI', float('nan')),
            'NMI': metrics.get('NMI', float('nan')),
            'valid_atoms': metrics.get('valid_atoms', 0),
            'total_atoms': metrics.get('total_atoms', 0),
            'final_total': log[-1]['total'] if log else float('nan'),
            'final_render': log[-1]['render'] if log else float('nan'),
            'final_coh': log[-1]['coh'] if log else float('nan'),
            'final_diff': log[-1]['diff'] if log else float('nan'),
        }
        for k, v in config.items():
            result[k] = v

        print(f"  >>> ARI={result['ARI']:.4f}  NMI={result['NMI']:.4f}  ({elapsed / 60:.1f} min)")
        sys.stdout.flush()
        return result

    except Exception as e:
        elapsed = time.time() - start
        print(f"  >>> FAILED after {elapsed / 60:.1f} min: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return {
            'trial_id': trial_id, 'phase': phase, 'tag': tag,
            'elapsed_min': f"{elapsed / 60:.1f}",
            'ARI': -999, 'NMI': -999, 'error': str(e),
            **config,
        }

    finally:
        # Aggressive cleanup between trials
        del atoms, field, log, metrics
        gc.collect()
        torch.cuda.empty_cache()


def save_results(results, phase):
    """Append results to CSV, overwriting for fresh start."""
    csv_path = BASE_DIR / f"phase{phase}_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        return

    fieldnames = list(results[0].keys())
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    print(f"\n[Saved] {csv_path}")


def print_summary(results):
    """Print sorted summary of results."""
    valid = [r for r in results if r.get('ARI', -999) > -998]
    if not valid:
        print("  No valid results to summarize.")
        return

    valid.sort(key=lambda r: r['ARI'], reverse=True)

    print(f"\n{'─' * 70}")
    print("TOP 10 by ARI:")
    print(
        f"{'Rank':<6} {'ARI':<8} {'NMI':<8} {'time':<8} Config"
    )
    print(f"{'─' * 70}")
    for i, r in enumerate(valid[:10]):
        config_str = " ".join(
            f"{k}={r[k]}"
            for k in ('w_met', 'w_vol', 'w_coh', 'w_direct', 'tau',
                       'diff_K', 'sinkhorn_eps', 'ent_weight')
            if k in r
        )
        print(
            f"{i + 1:<6} {r['ARI']:<8.4f} {r.get('NMI', 0):<8.4f} "
            f"{r.get('elapsed_min', '?'):<8} {config_str}"
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Grid Search for MetricAtom')
    parser.add_argument(
        '--phase', type=float, default=1, choices=[1, 1.5, 2, 3, 4, 4.5, 5],
        help='Search phase: 1=w_met×w_vol, 1.5=w_vol sweet, 2=w_coh×tau, '
             '3=diff_K, 4=w_direct×eps, 4.5=w_vol fine, 5=w_met×w_vol re-val'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print configs without running'
    )
    parser.add_argument(
        '--resume', action='store_true',
        help='Skip completed configs (check CSV for existing tags)'
    )
    args = parser.parse_args()

    # Handle phase 1.5/4.5 (float keys)
    if args.phase in (1.5, 4.5):
        phase_key = args.phase
    else:
        phase_key = int(args.phase)
    search_space = {
        1: PHASE1, 1.5: PHASE1_5, 2: PHASE2, 3: PHASE3,
        4: PHASE4, 4.5: PHASE4_5, 5: PHASE5,
    }[phase_key]
    # Phase 4+ uses direct cluster loss; 1-3 use InfoNCE
    use_direct = phase_key in (4, 4.5, 5)

    configs = build_configs(search_space)

    loss_type = "DirectCluster" if use_direct else "InfoNCE"
    print(f"Phase {args.phase} grid search: {len(configs)} configurations [{loss_type}]")
    print(f"Device: {DEVICE}  BF16: {BF16}")
    print(f"Each trial: 64×64, 500 epochs, quick mode (~3 min each)")
    print(f"Estimated total: ~{len(configs) * 3} min")

    if args.dry_run:
        print("\nConfigurations:")
        for i, cfg in enumerate(configs):
            print(f"  [{i + 1:2d}] {config_to_tag(cfg)}")
        return

    # Load existing results for resume
    completed_tags = set()
    csv_path = BASE_DIR / f"phase{args.phase}_results.csv"
    if args.resume and csv_path.exists():
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_tags.add(row.get('tag', ''))
        print(f"\nResume mode: {len(completed_tags)} already completed, "
              f"{len(configs) - len(completed_tags)} remaining")

    results = []
    for i, config in enumerate(configs):
        tag = config_to_tag(config)
        if tag in completed_tags:
            print(f"\n[Skipping] Trial {i + 1}/{len(configs)} — already completed: {tag}")
            continue

        trial_id = i + 1
        result = run_trial(config, trial_id, args.phase, use_direct_loss=use_direct)
        results.append(result)
        save_results(results, args.phase)

    print_summary(results)


if __name__ == '__main__':
    main()
