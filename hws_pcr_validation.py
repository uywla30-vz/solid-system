import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os

# --- Core HWS Components ---

def get_peak_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

class HWSPCRNetwork:
    """Manual HWS Network with per-layer coefficients and normalized Φ."""
    def __init__(self, k_harmonics=5, input_dim=2, hidden_dim=2, output_dim=1, seed=42):
        torch.manual_seed(seed)
        self.K = k_harmonics
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alphas1 = torch.randn(k_harmonics) * 0.1
        self.alphas2 = torch.randn(k_harmonics) * 0.1
        self.bias_h = torch.zeros(hidden_dim)
        self.bias_o = torch.zeros(output_dim)

        # Normalized Φ: 2π * (i/(Imax-1) + j/(Jmax-1))
        self.phi1 = torch.zeros(hidden_dim, input_dim)
        for i in range(hidden_dim):
            for j in range(input_dim):
                term_i = i / (hidden_dim - 1) if hidden_dim > 1 else 0
                term_j = j / (input_dim - 1) if input_dim > 1 else 0
                self.phi1[i, j] = 2 * np.pi * (term_i + term_j)

        self.phi2 = torch.zeros(output_dim, hidden_dim)
        for i in range(output_dim):
            for j in range(hidden_dim):
                term_i = i / (output_dim - 1) if output_dim > 1 else 0
                term_j = j / (hidden_dim - 1) if hidden_dim > 1 else 0
                self.phi2[i, j] = 2 * np.pi * (term_i + term_j)

    def get_weights(self):
        w1 = torch.zeros(self.hidden_dim, self.input_dim)
        w2 = torch.zeros(self.output_dim, self.hidden_dim)
        for k in range(1, self.K + 1):
            w1 += self.alphas1[k-1] * torch.cos(k * self.phi1)
            w2 += self.alphas2[k-1] * torch.cos(k * self.phi2)
        return w1, w2

    def forward(self, x):
        w1, w2 = self.get_weights()
        self.z1 = torch.matmul(x, w1.t()) + self.bias_h
        self.a1 = sigmoid(self.z1)
        self.z2 = torch.matmul(self.a1, w2.t()) + self.bias_o
        self.a2 = sigmoid(self.z2)
        return self.a2

# --- Training Algorithms ---

def train_pcr(model, x, y, iterations=10000, eta0=0.1, lam=0.001, tau=0.1):
    """Phase Coherence Rule (PCR) implementation without autograd."""
    losses = []
    prev_loss = None
    peak_ram = 0
    mean_cos1 = torch.stack([torch.mean(torch.cos(k * model.phi1)) for k in range(1, model.K + 1)])
    mean_cos2 = torch.stack([torch.mean(torch.cos(k * model.phi2)) for k in range(1, model.K + 1)])

    start_time = time.time()
    for t in range(iterations):
        peak_ram = max(peak_ram, get_peak_ram())
        out = model.forward(x)
        current_loss = 0.5 * torch.mean((out - y)**2).item()
        losses.append(current_loss)
        eta_t = eta0 / (1 + lam * t)

        if prev_loss is not None:
            delta_loss = current_loss - prev_loss
            tanh_term = torch.tanh(torch.tensor(delta_loss / tau))
            # PCR Update Rule
            model.alphas1 -= eta_t * tanh_term * torch.mean(x) * mean_cos1
            model.alphas2 -= eta_t * tanh_term * torch.mean(model.a1) * mean_cos2
            # Manual Bias Updates
            diff = out - y
            model.bias_o -= eta_t * torch.mean(diff, dim=0)
            _, w2 = model.get_weights()
            grad_b_h = torch.matmul(diff, w2) * d_sigmoid(model.z1)
            model.bias_h -= eta_t * torch.mean(grad_b_h, dim=0)
        prev_loss = current_loss
        if current_loss < 0.001: break
    return losses, peak_ram, time.time() - start_time

class HWSBPNetwork(nn.Module):
    """HWS Network for Backprop comparison using autograd."""
    def __init__(self, k_harmonics=5, input_dim=2, hidden_dim=2, output_dim=1, seed=42):
        super(HWSBPNetwork, self).__init__()
        torch.manual_seed(seed)
        self.K = k_harmonics
        self.alphas1 = nn.Parameter(torch.randn(k_harmonics) * 0.1)
        self.alphas2 = nn.Parameter(torch.randn(k_harmonics) * 0.1)
        self.bias_h = nn.Parameter(torch.zeros(hidden_dim))
        self.bias_o = nn.Parameter(torch.zeros(output_dim))

        phi1 = torch.zeros(hidden_dim, input_dim)
        for i in range(hidden_dim):
            for j in range(input_dim):
                term_i = i / (hidden_dim - 1) if hidden_dim > 1 else 0
                term_j = j / (input_dim - 1) if input_dim > 1 else 0
                phi1[i, j] = 2 * np.pi * (term_i + term_j)
        self.register_buffer('phi1', phi1)

        phi2 = torch.zeros(output_dim, hidden_dim)
        for i in range(output_dim):
            for j in range(hidden_dim):
                term_i = i / (output_dim - 1) if output_dim > 1 else 0
                term_j = j / (hidden_dim - 1) if hidden_dim > 1 else 0
                phi2[i, j] = 2 * np.pi * (term_i + term_j)
        self.register_buffer('phi2', phi2)

    def get_weights(self):
        w1 = torch.zeros(2, 2, device=self.alphas1.device)
        w2 = torch.zeros(1, 2, device=self.alphas2.device)
        for k in range(1, self.K + 1):
            w1 = w1 + self.alphas1[k-1] * torch.cos(k * self.phi1)
            w2 = w2 + self.alphas2[k-1] * torch.cos(k * self.phi2)
        return w1, w2

    def forward(self, x):
        w1, w2 = self.get_weights()
        h = torch.sigmoid(torch.matmul(x, w1.t()) + self.bias_h)
        o = torch.sigmoid(torch.matmul(h, w2.t()) + self.bias_o)
        return o

def train_bp(model, x, y, iterations=10000, lr=0.1):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    peak_ram = 0
    start_time = time.time()
    for t in range(iterations):
        peak_ram = max(peak_ram, get_peak_ram())
        optimizer.zero_grad()
        out = model(x)
        loss = 0.5 * torch.mean((out - y)**2)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if loss.item() < 0.001: break
    return losses, peak_ram, time.time() - start_time

# --- Execution ---

def run_experiment_4():
    x = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    print("--- Experiment 4A: Baseline PCR (K=5) ---")
    seeds = [42, 123, 777]
    for s in seeds:
        model = HWSPCRNetwork(k_harmonics=5, seed=s)
        losses, _, _ = train_pcr(model, x, y)
        out = model.forward(x)
        predicted = (out > 0.5).float()
        acc = (predicted == y).float().mean().item()
        print(f"Seed {s}: Accuracy {acc*100:.1f}%, Final Loss {losses[-1]:.6f}")

    print("\n--- Experiment 4C: PCR vs Backprop Head-to-Head (K=10) ---")
    # Best PCR
    model_pcr = HWSPCRNetwork(k_harmonics=10, seed=42)
    losses_pcr, ram_pcr, time_pcr = train_pcr(model_pcr, x, y, eta0=0.1, tau=0.01)
    out_pcr = model_pcr.forward(x)
    acc_pcr = ((out_pcr > 0.5).float() == y).float().mean().item()
    print(f"PCR: Accuracy {acc_pcr*100:.1f}%, Iterations {len(losses_pcr)}, Peak RAM {ram_pcr:.1f} MB, Time {time_pcr:.2f}s")

    # Backprop
    model_bp = HWSBPNetwork(k_harmonics=10, seed=42)
    losses_bp, ram_bp, time_bp = train_bp(model_bp, x, y)
    out_bp = model_bp(x)
    acc_bp = ((out_bp > 0.5).float() == y).float().mean().item()
    print(f"Backprop: Accuracy {acc_bp*100:.1f}%, Iterations {len(losses_bp)}, Peak RAM {ram_bp:.1f} MB, Time {time_bp:.2f}s")

    # Micro-benchmark
    start = time.time()
    for _ in range(1000): model_pcr.get_weights()
    syn_time = (time.time() - start) / 1000
    w1, w2 = model_pcr.get_weights()
    start = time.time()
    for _ in range(1000): _ = w1.clone(), w2.clone()
    load_time = (time.time() - start) / 1000
    print(f"\nWeight Synthesis (K=10): {syn_time*1e6:.2f}µs vs Loading: {load_time*1e6:.2f}µs")

    # Save Plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses_pcr, label='PCR (K=10)')
    plt.plot(losses_bp, label='Backprop (K=10)')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Experiment 4C: PCR vs Backprop (XOR)')
    plt.legend()
    plt.savefig('experiment4_results.png')
    print("\nResults plot saved as experiment4_results.png")

if __name__ == "__main__":
    run_experiment_4()
