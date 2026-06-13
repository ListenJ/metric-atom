"""
Homeostatic plasticity regularizer for MetricAtom.

Motivation (bias-reduced): training self-organizing atoms shows large seed
variance (sigma=0.39 ARI).  Biological recurrent circuits use homeostatic
mechanisms to keep neurons in a useful dynamic range and avoid saturation or
quiescence.  This module adds lightweight, differentiable regularization that
pushes atom existence probabilities and per-view contributions toward a target
range, reducing runaway suppression/activation that can trap bad seeds.

Scope: small additive loss term, no new architecture, no external priors.
"""

import torch
import torch.nn.functional as F


def existence_homeostasis(existence_probs, target_mean=0.5, target_std=0.25):
    """
    Push the population of atom existence probabilities toward a target mean
    and keep its spread bounded.

    Args:
        existence_probs: (N,) tensor of eps_i in (0, 1).
        target_mean: desired population mean of existence probabilities.
        target_std: desired approximate population std; we penalize excess.

    Returns:
        scalar loss
    """
    if existence_probs.numel() == 0:
        return torch.tensor(0.0, device=existence_probs.device)

    mean = existence_probs.mean()
    # Penalize both low and high population mean relative to target.
    loss_mean = F.mse_loss(mean, torch.tensor(target_mean,
                                              device=existence_probs.device,
                                              dtype=existence_probs.dtype))

    # Soft upper bound on std to prevent all atoms collapsing to same eps.
    std = existence_probs.std(unbiased=False)
    loss_std = F.relu(std - target_std)

    return loss_mean + loss_std


def contribution_homeostasis(per_atom_contrib, target_log_density=0.0,
                             max_log_ratio=1.0):
    """
    Push per-atom total density contribution toward a common log target and
    bound log-ratio disparity.  This mirrors synaptic/homeostatic scaling that
    prevents a few atoms from dominating or vanishing.

    Args:
        per_atom_contrib: (N,) total density contribution per atom (>=0).
        target_log_density: log target for mean contribution.
        max_log_ratio: soft upper bound on log(max/contrib_mean).

    Returns:
        scalar loss
    """
    if per_atom_contrib.numel() == 0:
        return torch.tensor(0.0, device=per_atom_contrib.device)

    eps = 1e-8
    mean_contrib = per_atom_contrib.mean()
    loss_mean = F.mse_loss(torch.log(mean_contrib + eps),
                           torch.tensor(target_log_density,
                                        device=per_atom_contrib.device,
                                        dtype=per_atom_contrib.dtype))

    # Soft upper bound on max/mean ratio to fight oligarch atoms.
    max_contrib = per_atom_contrib.max()
    log_ratio = torch.log((max_contrib + eps) / (mean_contrib + eps))
    loss_ratio = F.relu(log_ratio - max_log_ratio)

    return loss_mean + loss_ratio


def homeostatic_loss(atoms, per_atom_contrib=None,
                     target_mean=0.5, target_std=0.25,
                     target_log_density=0.0, max_log_ratio=1.0):
    """
    Combined homeostatic loss on atom existence and optional contributions.

    Args:
        atoms: list of atoms with .existence_prob property.
        per_atom_contrib: optional (N,) per-atom density contribution.
        remaining args: see above.

    Returns:
        scalar loss
    """
    if len(atoms) == 0:
        return torch.tensor(0.0,
                            device=next(iter(atoms)).position.device
                            if atoms else 'cpu')

    device = atoms[0].position.device
    existence_probs = torch.stack([a.existence_prob for a in atoms])
    loss = existence_homeostasis(existence_probs, target_mean, target_std)

    if per_atom_contrib is not None:
        loss = loss + contribution_homeostasis(
            per_atom_contrib, target_log_density, max_log_ratio)

    return loss
