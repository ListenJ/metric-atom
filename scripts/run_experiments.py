"""
Batch experiment runner for MetricAtom.
Saves all results to outputs/exp/results.csv
"""
import sys, os, csv, time, gc, traceback, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from pathlib import Path
from train_2d import train_scene

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BF16 = DEVICE == 'cuda' and torch.cuda.is_bf16_supported()

EXPERIMENTS = [
    # (name, {overrides})
    ('3obj_64x64', {
        'H': 64, 'W': 64, 'num_atoms': 100, 'num_epochs': 500,
        'num_objects': 3, 'seed': 42,
        'w_met': 0.005, 'w_vol': 0.1,
        'use_direct_loss': True, 'w_direct': 2.0,
        'sinkhorn_eps': 0.5, 'sinkhorn_iters': 50,
        'diff_K': 5, 'quick_mode': True,
    }),
    ('seed123', {
        'H': 64, 'W': 64, 'num_atoms': 100, 'num_epochs': 500,
        'num_objects': 2, 'seed': 123,
        'w_met': 0.005, 'w_vol': 0.1,
        'use_direct_loss': True, 'w_direct': 2.0,
        'sinkhorn_eps': 0.5, 'sinkhorn_iters': 50,
        'diff_K': 5, 'quick_mode': True,
    }),
    ('seed456', {
        'H': 64, 'W': 64, 'num_atoms': 100, 'num_epochs': 500,
        'num_objects': 2, 'seed': 456,
        'w_met': 0.005, 'w_vol': 0.1,
        'use_direct_loss': True, 'w_direct': 2.0,
        'sinkhorn_eps': 0.5, 'sinkhorn_iters': 50,
        'diff_K': 5, 'quick_mode': True,
    }),
    ('128x128', {
        'H': 128, 'W': 128, 'num_atoms': 150, 'num_epochs': 600,
        'num_objects': 2, 'seed': 42,
        'w_met': 0.005, 'w_vol': 0.1,
        'use_direct_loss': True, 'w_direct': 2.0,
        'sinkhorn_eps': 0.5, 'sinkhorn_iters': 50,
        'diff_K': 5, 'quick_mode': True,
        'render_chunk_size': 1024,
        'num_samples': 64,
    }),
]

BASE_DIR = Path('outputs/exp')
RESULTS_CSV = BASE_DIR / 'results.csv'

def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for name, params in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*60}")
        sys.stdout.flush()

        output_dir = BASE_DIR / name / 'output'
        params['output_dir'] = str(output_dir)
        params['device'] = DEVICE
        params['bf16'] = BF16

        start = time.time()
        try:
            atoms, field, log, metrics = train_scene(**params)
            elapsed = time.time() - start

            row = {
                'name': name,
                'ARI': metrics.get('ARI', float('nan')),
                'NMI': metrics.get('NMI', float('nan')),
                'valid_atoms': metrics.get('valid_atoms', 0),
                'total_atoms': metrics.get('total_atoms', 0),
                'elapsed_min': f"{elapsed/60:.1f}",
                'params': str({k: v for k, v in params.items() if k != 'device' and k != 'output_dir' and k != 'bf16'})
            }
            results.append(row)

            print(f"  >>> ARI={row['ARI']:.4f}  NMI={row['NMI']:.4f}  valid={row['valid_atoms']}/{row['total_atoms']}  ({row['elapsed_min']} min)")

        except Exception as e:
            elapsed = time.time() - start
            print(f"  >>> FAILED after {elapsed/60:.1f} min: {e}")
            traceback.print_exc()
            results.append({
                'name': name, 'ARI': float('nan'), 'NMI': float('nan'),
                'valid_atoms': 0, 'total_atoms': 0, 'elapsed_min': f"{elapsed/60:.1f}",
                'params': str(params.get('num_objects', 2))
            })

        sys.stdout.flush()
        gc.collect()
        torch.cuda.empty_cache()

        # Save incremental results
        with open(RESULTS_CSV, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['name','ARI','NMI','valid_atoms','total_atoms','elapsed_min','params'])
            w.writeheader()
            w.writerows(results)

    # Print summary
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: x.get('ARI', -1), reverse=True):
        print(f"  {r['name']:20s}  ARI={r['ARI']:.4f}  NMI={r.get('NMI', 0):.4f}  valid={r['valid_atoms']}/{r['total_atoms']}")

if __name__ == '__main__':
    main()
