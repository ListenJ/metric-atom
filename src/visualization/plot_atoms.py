import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_atom_scatter(atoms, H, W, step, output_path):
    """
    快速原子散点图 — 用于训练过程中的快速检查。
    """
    mus = np.stack([a.position.detach().cpu().numpy() for a in atoms])
    colors = np.stack([a._color.detach().cpu().numpy().clip(0, 1) for a in atoms])
    eps_vals = np.array([a.existence_prob.detach().cpu().item() for a in atoms])
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.scatter(mus[:, 0] * W, mus[:, 1] * H,
               c=colors, s=eps_vals * 40 + 5, alpha=0.7,
               edgecolors='black', linewidth=0.3)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(f'Atom positions (step {step})')
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path / f'atoms_{step:04d}.png', dpi=80)
    plt.close(fig)
