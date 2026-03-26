import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

class HWSPCRV2Network:
    """HWS Network with PCR-V2 features."""
    def __init__(self, k_harmonics=5, layer_dims=[2, 2, 1], seed=42):
        torch.manual_seed(seed)
        self.K = k_harmonics
        self.dims = layer_dims
        self.num_layers = len(layer_dims) - 1
        primes = [2, 3, 5, 7, 11]
        # PDI Initialization
        self.alphas = [torch.randn(k_harmonics) * 0.1 * primes[l] for l in range(self.num_layers)]
        self.biases = [torch.zeros(dim) for dim in layer_dims[1:]]
        self.phis = []
        for l in range(self.num_layers):
            i_max, j_max = self.dims[l+1], self.dims[l]
            phi = torch.zeros(i_max, j_max)
            d_theta = np.pi / i_max
            d_phi = 2 * np.pi / j_max
            for i in range(i_max):
                for j in range(j_max):
                    # Corrected Spherical Mapping with 0.5 offset
                    if i_max > 1 and j_max > 1:
                        phi[i, j] = np.sin((i + 0.5) * d_theta) * np.cos((j + 0.5) * d_phi)
                    elif i_max == 1:
                        phi[i, j] = np.cos((j + 0.5) * d_phi)
                    elif j_max == 1:
                        phi[i, j] = np.sin((i + 0.5) * d_theta)
            self.phis.append(phi)

    def get_weights(self):
        ws = []
        for l in range(self.num_layers):
            w = torch.zeros(self.dims[l+1], self.dims[l])
            for k in range(1, self.K + 1):
                w += self.alphas[l][k-1] * torch.cos(k * self.phis[l])
            ws.append(w)
        return ws

    def forward(self, x):
        ws = self.get_weights()
        self.activations = [x]
        a = x
        for l in range(self.num_layers):
            a = sigmoid(torch.matmul(a, ws[l].t()) + self.biases[l])
            self.activations.append(a)
        return a

def train_pcr_v2(model, x, y, iterations=10000, eta0=0.1, lam=0.001, tau0=1.0, gamma=None):
    if gamma is None:
        gamma = -np.log(0.01) / (iterations / 2)
    losses = []
    cos_k_phis = [[torch.cos(k * model.phis[l]) for k in range(1, model.K + 1)] for l in range(model.num_layers)]
    for t in range(iterations):
        out = model.forward(x)
        loss = 0.5 * torch.mean((y - out)**2)
        losses.append(loss.item())
        tau_t = tau0 * np.exp(-gamma * t)
        eta_t = eta0 / (1 + lam * t)
        delta_global = torch.tanh((y - out) / tau_t)
        for l in range(model.num_layers):
            prod_mean_abs = 1.0
            for i in range(l + 1, model.num_layers):
                prod_mean_abs *= torch.mean(torch.abs(model.alphas[i]))
            delta_L = delta_global * prod_mean_abs
            act_prev = model.activations[l]
            for k in range(1, model.K + 1):
                cos_k_phi = cos_k_phis[l][k-1]
                # Update term calculated as mean over the layer and batch
                update = torch.mean(delta_L.unsqueeze(2) * act_prev.unsqueeze(1) * cos_k_phi.unsqueeze(0))
                model.alphas[l][k-1] -= eta_t * update
            model.biases[l] -= eta_t * torch.mean(delta_L, dim=0)
        if loss.item() < 0.001: break
    return losses

if __name__ == "__main__":
    x = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    seeds = [42, 123, 777]
    print("--- Experiment 5A: PCR-V2 on XOR (Final Report) ---")
    for K in [5, 10]:
        print(f"\nK={K}:")
        for s in seeds:
            model = HWSPCRV2Network(k_harmonics=K, seed=s)
            losses = train_pcr_v2(model, x, y, iterations=10000, eta0=0.8)
            acc = ((model.forward(x) > 0.5).float() == y).float().mean().item()
            print(f"Seed {s:3d}: Accuracy {acc*100:5.1f}%, Final Loss {losses[-1]:.6f}")
