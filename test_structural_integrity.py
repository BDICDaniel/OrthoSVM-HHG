"""
Unit Test: Structural Integrity, Mathematical Orthogonality, and Zero Leakage Verification.
"""

import sys
import numpy as np
import torch
import torch.nn as nn

import dataset
import hyperdual_pro

def test_structural_beauty():
    print("=" * 70)
    print(" 1. VERIFYING MATHEMATICAL ORTHOGONALITY IN HILBERT SPACE")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    d_o = 24
    N = 100
    module = hyperdual_pro.StructuralOrthogonalDualExperts(d_o=d_o, K=3).to(device)
    
    # Random dummy representations
    r = torch.randn(N, d_o, device=device)
    prob, (score_r, score_nr, f_r_perp, f_nr_perp, geom_margin) = module(r)
    
    # Compute dot products under the learned Riemannian metric M = U^T U + 1e-4 I
    M, P_nr, Pi_perp = module.core.get_projectors()
    idem_err = torch.norm(Pi_perp @ Pi_perp - Pi_perp).item()

    # Check exact orthogonality <f_r_perp, f_nr>_M == 0 for every sample
    residual_inner_product = (f_r_perp * (f_nr_perp @ M)).sum(dim=-1)
    max_residual = residual_inner_product.abs().max().item()
    mean_residual = residual_inner_product.abs().mean().item()
    
    print(f"Projector Idempotence ||Pi_perp^2 - Pi_perp||_F : {idem_err:.2e}")
    print(f"Orthogonality residual <f_r_perp, f_nr>_M:")
    print(f"  Max absolute residual  : {max_residual:.2e}")
    print(f"  Mean absolute residual : {mean_residual:.2e}")
    assert idem_err < 1e-4, f"Idempotence check failed! Error: {idem_err}"
    assert max_residual < 1e-3, f"Orthogonality check failed! Residual: {max_residual}"
    print("  -> Mathematical Hilbert Space Subspace Projection strictly verified!\n")

    print("=" * 70)
    print(" 2. VERIFYING SVM MAXIMUM-MARGIN GRADIENT DYNAMICS")
    print("=" * 70)
    # Test backward pass through margin
    loss = (prob - 0.5).pow(2).mean()
    loss.backward()
    
    grad_norm_U = module.core.metric_U.grad.norm().item()
    grad_norm_wr = module.w_risk.weight.grad.norm().item()
    grad_norm_wnr = module.w_nonrisk.weight.grad.norm().item()
    grad_norm_gamma = module.gamma.grad.abs().item()
    
    print(f"Gradient norms:")
    print(f"  Metric U grad norm     : {grad_norm_U:.6f}")
    print(f"  SVM w_risk grad norm   : {grad_norm_wr:.6f}")
    print(f"  SVM w_nonrisk grad norm: {grad_norm_wnr:.6f}")
    print(f"  Margin scale gamma grad: {grad_norm_gamma:.6f}")
    assert grad_norm_U > 0 and grad_norm_wr > 0 and grad_norm_wnr > 0, "Gradients are zero!"
    print("  -> End-to-end differentiable structural learning verified!\n")

    print("=" * 70)
    print(" 3. VERIFYING DATA LEAKAGE ISOLATION")
    print("=" * 70)
    data = dataset.load_bphd_data()
    train_set = set(data['train_idx'])
    test_set = set(data['test_idx'])
    valid_set = set(data['valid_idx'])
    
    print(f"Clean Train Set count : {len(train_set)}")
    print(f"Clean Valid Set count : {len(valid_set)}")
    print(f"Clean Test Set count  : {len(test_set)}")
    print(f"Train & Test Overlap  : {len(train_set & test_set)}")
    print(f"Train & Valid Overlap : {len(train_set & valid_set)}")
    print(f"Valid & Test Overlap  : {len(valid_set & test_set)}")
    
    assert len(train_set & test_set) == 0, "Data leakage detected: train and test overlap!"
    assert len(train_set & valid_set) == 0, "Data leakage detected: train and valid overlap!"
    assert len(valid_set & test_set) == 0, "Data leakage detected: valid and test overlap!"
    print("  -> Strict 0-leakage sample quarantine verified!\n")
    print("ALL STRUCTURAL INTEGRITY TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_structural_beauty()
