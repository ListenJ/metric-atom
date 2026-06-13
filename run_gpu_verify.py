"""本地GPU 2D训练验证 — 适配4GB VRAM (RTX 3050 Ti)

目标：确认2D训练在GPU上完美运行，无NaN、无崩溃、无错误。
配置：32×32, 30 atoms, BF16, ~4GB VRAM
"""
import torch
import warnings
warnings.filterwarnings('ignore')

from train_2d import train_scene

def run_gpu_train(name, epochs=200, resolution=32, atoms=30, **kwargs):
    torch.manual_seed(42)
    print(f"\n{'='*60}")
    print(f"GPU TRAIN: {name}")
    print(f"{'='*60}")
    print(f"Config: {resolution}×{resolution} | {atoms} atoms | {epochs} epochs | BF16")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    atoms_out, mf, losses, metrics = train_scene(
        H=resolution, W=resolution,
        num_atoms=atoms,
        num_epochs=epochs,
        num_views=8,
        lr=1e-3,
        device='cuda',
        output_dir=f'outputs/gpu_verify_{name}',
        bf16=True,          # 3050 Ti supports BF16
        fp16=False,
        num_samples=32,
        seed=42,
        w_met=0.01,
        w_vol=1.0,
        w_tc=2.0,
        w_pos=5.0,
        w_selforg=1.0,
        w_predict=1.0,
        state_alpha=0.3,
        mask_ratio=0.3,
        diff_K=5,
        chunk_size=256,     # 32*32 = 1024 rays, 256 chunk = 4 chunks
        atom_chunk_size=10, # 4GB safe
        metric_batch_size=128,
        phase1_epochs=epochs // 3,  # Phase 1 = 1/3 of total
        w_vol_p1=5.0,
        w_tc_p1=10.0,
        no_diffusion=True,  # save VRAM
        **kwargs
    )
    
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"\nPeak VRAM: {peak_mb:.0f} MB")
    print(f"Result: {len(atoms_out)} atoms | ARI={metrics.get('ARI', 'N/A')} | NMI={metrics.get('NMI', 'N/A')}")
    
    # Check for NaN in losses
    has_nan = any(torch.isnan(torch.tensor(l['total'])) for l in losses)
    print(f"Loss NaN detected: {has_nan}")
    
    # Check metric field for NaN
    with torch.no_grad():
        tr = mf.trace()
        metric_nan = torch.isnan(tr).any().item()
    print(f"Metric field NaN: {metric_nan}")
    
    return not has_nan and not metric_nan and metrics.get('ARI', 0) is not None


if __name__ == '__main__':
    print("MetricAtom 2D GPU Verification")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    
    # Test 1: Standard scene (different colors), Cholesky
    ok1 = run_gpu_train("std_cholesky_200ep", epochs=200, parametrization='cholesky')
    
    # Test 2: Same-color scene, Cholesky (EXT-4 validation)
    ok2 = run_gpu_train("samecolor_200ep", epochs=200, same_color=True, parametrization='cholesky')
    
    # Test 3: Standard scene, matrix_exp (EXT-1 validation) — shorter due to speed
    ok3 = run_gpu_train("std_matrixexp_100ep", epochs=100, parametrization='matrix_exp')
    
    print(f"\n{'='*60}")
    print("GPU VERIFICATION COMPLETE")
    print(f"Standard+Cholesky (200ep): {'PASS ✓' if ok1 else 'FAIL ✗'}")
    print(f"Same-color+Cholesky (200ep): {'PASS ✓' if ok2 else 'FAIL ✗'}")
    print(f"Standard+matrix_exp (100ep): {'PASS ✓' if ok3 else 'FAIL ✗'}")
    print(f"{'='*60}")
    
    if ok1 and ok2 and ok3:
        print("\n✅ ALL GPU TESTS PASSED — 2D training is stable")
    else:
        print("\n❌ SOME TESTS FAILED — check logs above")
