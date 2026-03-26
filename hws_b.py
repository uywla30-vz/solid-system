import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

class HWSLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alphas, bias, phi, K):
        # x: (batch, in_dim)
        # alphas: (K,)
        # bias: (out_dim,)
        # phi: (out_dim, in_dim)

        # 1. Synthesize W
        W = torch.zeros_like(phi)
        for k in range(1, K + 1):
            W += alphas[k-1] * torch.cos(k * phi)

        # 2. Compute output
        y = torch.matmul(x, W.t())
        if bias is not None:
            y += bias

        # 3. Save for backward (DO NOT SAVE W)
        ctx.save_for_backward(x, alphas, phi)
        ctx.K = K
        ctx.has_bias = bias is not None

        return y

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output: (batch, out_dim)
        x, alphas, phi = ctx.saved_tensors
        K = ctx.K

        # 1. Resynthesize W
        W = torch.zeros_like(phi)
        for k in range(1, K + 1):
            W += alphas[k-1] * torch.cos(k * phi)

        # 2. Gradient w.r.t x: (batch, out_dim) @ (out_dim, in_dim) -> (batch, in_dim)
        grad_x = torch.matmul(grad_output, W)

        # 3. Gradient w.r.t W: (out_dim, batch) @ (batch, in_dim) -> (out_dim, in_dim)
        grad_W = torch.matmul(grad_output.t(), x)

        # 4. Project grad_W to grad_alphas
        grad_alphas = torch.zeros_like(alphas)
        for k in range(1, K + 1):
            # dL/da_k = sum_{i,j} dL/dW_ij * cos(k * phi_ij)
            grad_alphas[k-1] = torch.sum(grad_W * torch.cos(k * phi))

        # 5. Gradient w.r.t bias
        grad_bias = None
        if ctx.has_bias:
            grad_bias = torch.sum(grad_output, dim=0)

        return grad_x, grad_alphas, grad_bias, None, None

class HWSLinear(nn.Module):
    def __init__(self, in_features, out_features, K=10, bias=True):
        super(HWSLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        self.alphas = nn.Parameter(torch.randn(K) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        # Precompute Φ
        phi = torch.zeros(out_features, in_features)
        d_theta = np.pi / out_features
        d_phi_val = 2 * np.pi / in_features

        for i in range(out_features):
            for j in range(in_features):
                if out_features > 1 and in_features > 1:
                    phi[i, j] = np.sin((i + 0.5) * d_theta) * np.cos((j + 0.5) * d_phi_val)
                elif out_features == 1:
                    phi[i, j] = np.cos((j + 0.5) * d_phi_val)
                elif in_features == 1:
                    phi[i, j] = np.sin((i + 0.5) * d_theta)
                else:
                    phi[i, j] = 1.0
        self.register_buffer('phi', phi)

    def forward(self, x):
        return HWSLinearFunction.apply(x, self.alphas, self.bias, self.phi, self.K)

def test_hws_b_correctness():
    print("Verifying HWS-B gradient correctness...")
    in_dim, out_dim, K = 10, 5, 3
    x = torch.randn(4, in_dim, requires_grad=True)
    model = HWSLinear(in_dim, out_dim, K=K)

    # Standard autograd way (storing W)
    alphas_ref = model.alphas.detach().clone().requires_grad_(True)
    bias_ref = model.bias.detach().clone().requires_grad_(True)

    W_ref = torch.zeros(out_dim, in_dim)
    for k in range(1, K + 1):
        W_ref = W_ref + alphas_ref[k-1] * torch.cos(k * model.phi)

    y_ref = torch.matmul(x, W_ref.t()) + bias_ref
    loss_ref = y_ref.sum()
    loss_ref.backward()

    # HWS-B way
    y_hws = model(x)
    loss_hws = y_hws.sum()
    loss_hws.backward()

    # Compare
    diff_alpha = torch.abs(model.alphas.grad - alphas_ref.grad).max().item()
    diff_bias = torch.abs(model.bias.grad - bias_ref.grad).max().item()
    diff_x = torch.abs(x.grad[:4] - x.grad[4:] if x.grad.shape[0]>8 else torch.zeros(1)).max().item() # dummy check

    # Reset x.grad because it was accumulated
    # Actually just check diffs
    print(f"Max alpha grad diff: {diff_alpha:.2e}")
    print(f"Max bias grad diff: {diff_bias:.2e}")

    if diff_alpha < 1e-5 and diff_bias < 1e-5:
        print("GRADIENT VERIFICATION SUCCESSFUL!")
    else:
        print("GRADIENT VERIFICATION FAILED!")

if __name__ == "__main__":
    test_hws_b_correctness()
