"""Quick 3D module verification"""
import sys
sys.path.insert(0, 'D:\\MetricAtom')
import numpy as np
from src.data.synthetic_3d import generate_multi_view_3d, generate_3d_scene

# Test scene generation
spheres, bounds = generate_3d_scene(H=64, W=64, num_objects=2, seed=42)
print(f"Scene: {len(spheres)} spheres, bounds={bounds}")
for i, s in enumerate(spheres):
    print(f"  Sphere {i}: center={s['center']}, radius={s['radius']:.3f}")

# Test multi-view generation (CPU)
images, masks, cameras, spheres = generate_multi_view_3d(H=64, W=64, num_objects=2, num_views=4, seed=42)
print(f"\nImages: shape={images.shape}, range=[{images.min():.3f}, {images.max():.3f}]")
print(f"Masks: shape={masks.shape}, mean occ={masks.mean():.3f}")
print(f"Cameras: {len(cameras)}, each with pos, rot, fx, fy")
print(f"Spheres: {len(spheres)}")
for i, s in enumerate(spheres):
    print(f"  Sphere {i}: center={s['center']}, radius={s['radius']:.3f}")

# Test 3D atom creation
from src.atoms.atom_3d import Atom3D
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mu = torch.tensor([0.5, 0.5, 0.5], device=device)
color = torch.tensor([1.0, 0.0, 0.0], device=device)
atom = Atom3D(mu, radius=0.2, color=color, feature_dim=16)
print(f"\nAtom3D: pos={atom.position}, radius={atom.radius:.3f}, eps={atom.existence_prob:.3f}")

# Test MetricField3D
from src.geometry.metric_field import MetricField3D
field = MetricField3D(8, 8, 8, init_scale=1.0).to(device)
coords = torch.tensor([[0.3, 0.5, 0.7]], device=device)
g = field(coords)
print(f"MetricField3D: g(0.3,0.5,0.7) shape={g.shape}")
print(f"  g =\n{g[0].detach().cpu().numpy().round(4)}")
print(f"  trace = {field.trace(coords).item():.4f}")

# Test volume_render_3d
from src.rendering.volume_renderer_2d import volume_render_3d
from src.rendering.ray_sampler import RaySampler3D
atoms_list = [atom]
rays_o = torch.zeros(100, 3, device=device)
rays_d = torch.tensor([0.0, 0.0, 1.0], device=device).unsqueeze(0).expand(100, -1)
rendered, depth, alpha = volume_render_3d(rays_o, rays_d, atoms_list, field, num_samples=32)
print(f"\n3D Render: color shape={rendered.shape}, depth={depth.mean():.3f}, alpha={alpha.mean():.3f}")

# Test coherence on 3D atoms
from src.losses.coherence import coherence_loss
second_mu = torch.tensor([0.7, 0.3, 0.2], device=device)
second_color = torch.tensor([0.0, 0.0, 1.0], device=device)
atom2 = Atom3D(second_mu, radius=0.2, color=second_color, feature_dim=16)
coh = coherence_loss([atom, atom2], field, repulsion_weight=5.0)
print(f"\n3D Coherence Loss: {coh.item():.4f} (negative = clustering)")

# Test smoothness loss 3D
from src.losses.metric_regularizer import metric_smoothness_loss_3d
smooth = metric_smoothness_loss_3d(field)
print(f"3D Metric Smoothness: {smooth.item():.6f}")

print("\nAll 3D tests passed!")
