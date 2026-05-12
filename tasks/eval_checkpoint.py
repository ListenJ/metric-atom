"""Quick evaluation of latest checkpoint for coverage and ARI."""
import sys, torch, numpy as np
sys.path.insert(0, 'D:\\MetricAtom')

from src.data.synthetic_2d import generate_multi_view, get_occupancy
from src.visualization.plot_metric import evaluate_clustering
from src.atoms.atom_2d import Atom2D
from src.geometry.metric_field import MetricField2D

device = 'cuda'
H = W = 64

# Load checkpoint
ckpt = torch.load('D:\\MetricAtom\\outputs\\2d_64x64_bf16\\checkpoint.pt', map_location=device)
print(f'Loaded checkpoint ({len(ckpt["losses_log"])} entries)')
for ld in ckpt['losses_log'][-3:]:
    print(f"  E{ld['epoch']}: T={ld['total']:.3f} R={ld['render']:.3f} V={ld['vol']:.3f} C={ld['coh']:.3f} P={ld['pos']:.4f}")

# Rebuild
metric_field = MetricField2D(H, W, init_scale=1.0).to(device)
metric_field.load_state_dict(ckpt['metric_field'])

atoms = []
fake_mu = torch.tensor([0.5, 0.5], device=device)
for sd in ckpt['atoms']:
    a = Atom2D(fake_mu, 0.1, torch.rand(3, device=device), feature_dim=16)
    a.load_state_dict(sd)
    atoms.append(a)
print(f'Atoms: {len(atoms)}')

# Coverage
images_np, masks_np, _ = generate_multi_view(H, W, num_objects=2, num_views=8, seed=42)
mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
mask_v0 = masks_np[0]; K = mask_v0.shape[-1]
obj_atoms = 0
for mu in mus:
    px = int(np.clip(mu[0] * W, 0, W-1))
    py = int(np.clip(mu[1] * H, 0, H-1))
    if any(mask_v0[py, px, k] > 0.5 for k in range(K)):
        obj_atoms += 1
print(f'Coverage: {obj_atoms}/{len(atoms)} = {100*obj_atoms/len(atoms):.1f}%')

# Evaluation
metrics = evaluate_clustering(atoms, masks_np[np.newaxis], H, W)
print(f'ARI={metrics["ARI"]} NMI={metrics["NMI"]} valid={metrics["valid_atoms"]}/{metrics["total_atoms"]}')

# Trace
trace = metric_field.trace().detach().cpu().numpy()
print(f'Trace: mean={trace.mean():.3f} min={trace.min():.3f} max={trace.max():.3f}')
print(f'Loss trend (first 5):')
for ld in ckpt['losses_log'][:5]:
    print(f"  E{ld['epoch']}: P={ld['pos']:.4f}")
