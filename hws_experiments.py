import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import psutil
import os

# --- Core HWS Components ---

def cantor_pairing(i, j):
    """Φ(i,j) = Cantor pairing function: (i+j)(i+j+1)/2 + j"""
    return (i + j) * (i + j + 1) // 2 + j

class HWSNetwork(nn.Module):
    def __init__(self, k_harmonics=5):
        super(HWSNetwork, self).__init__()
        self.k_harmonics = k_harmonics
        # α coefficients: initialized from standard normal scaled by 0.1
        self.alphas = nn.Parameter(torch.randn(k_harmonics) * 0.1)
        # Biases: conventional learnable scalars
        self.bias_h = nn.Parameter(torch.zeros(2))
        self.bias_o = nn.Parameter(torch.zeros(1))

        # Neuron IDs: 2 input (0,1), 2 hidden (2,3), 1 output (4)
        input_ids, hidden_ids, output_ids = [0, 1], [2, 3], [4]

        # Precompute Φ(i,j) for spatial encoding
        phi_l1 = [[cantor_pairing(i, j) for i in input_ids] for j in hidden_ids]
        self.register_buffer('phi_l1', torch.tensor(phi_l1, dtype=torch.float32))

        phi_l2 = [[cantor_pairing(i, j) for i in hidden_ids] for j in output_ids]
        self.register_buffer('phi_l2', torch.tensor(phi_l2, dtype=torch.float32))

    def get_weights(self):
        """Ψ(i,j) = Σ α_k · cos(k · Φ(i,j)) for k=1 to K"""
        w1 = torch.zeros_like(self.phi_l1)
        w2 = torch.zeros_like(self.phi_l2)
        for k in range(1, self.k_harmonics + 1):
            cos_k_phi1 = torch.cos(k * self.phi_l1)
            cos_k_phi2 = torch.cos(k * self.phi_l2)
            w1 = w1 + self.alphas[k-1] * cos_k_phi1
            w2 = w2 + self.alphas[k-1] * cos_k_phi2
        return w1, w2

    def forward(self, x):
        w1, w2 = self.get_weights()
        h = torch.sigmoid(torch.matmul(x, w1.t()) + self.bias_h)
        o = torch.sigmoid(torch.matmul(h, w2.t()) + self.bias_o)
        return o

# --- Utilities ---

def get_peak_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # MB

def benchmark_weight_gen(model, trials=1000):
    """Measure Ψ synthesis time vs loading equivalent static weights."""
    # Synthesis time
    start = time.time()
    for _ in range(trials):
        _ = model.get_weights()
    synthesis_time = (time.time() - start) / trials

    # Emulate static loading time (loading same sized tensors from memory)
    w1, w2 = model.get_weights()
    start = time.time()
    for _ in range(trials):
        _ = w1.clone(), w2.clone()
    static_load_time = (time.time() - start) / trials

    return synthesis_time, static_load_time

# --- Training Algorithms ---

def train_gemini_rule(model, x, y, iterations=10000, lr_alpha=0.01, lr_bias=0.1):
    criterion = nn.MSELoss()
    optimizer_bias = optim.SGD([model.bias_h, model.bias_o], lr=lr_bias)

    losses = []
    prev_loss = None
    all_phis = torch.cat([model.phi_l1.flatten(), model.phi_l2.flatten()])
    # cos(Phase_local) for α_k = mean over all (i,j) of cos(k · Φ(i,j))
    mean_cos_k_phi = torch.stack([torch.mean(torch.cos(k * all_phis)) for k in range(1, model.k_harmonics + 1)])

    peak_ram = 0
    for i in range(iterations):
        peak_ram = max(peak_ram, get_peak_ram())
        outputs = model(x)
        loss = criterion(outputs, y)
        current_loss = loss.item()
        losses.append(current_loss)

        optimizer_bias.zero_grad()
        loss.backward()
        model.alphas.grad = None # Exclude alphas from backprop
        optimizer_bias.step()

        if prev_loss is not None:
            delta_loss = current_loss - prev_loss
            sign = 1.0 if delta_loss > 0 else (-1.0 if delta_loss < 0 else 0.0)
            with torch.no_grad():
                # Δα_k = η · Sign(δLoss) · cos(Phase_local)
                model.alphas.data += lr_alpha * sign * mean_cos_k_phi
        prev_loss = current_loss

    return losses, peak_ram

def train_backprop(model, x, y, iterations=10000, lr=0.1, optimizer_type='SGD'):
    criterion = nn.MSELoss()
    if optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr)

    losses = []
    peak_ram = 0
    for i in range(iterations):
        peak_ram = max(peak_ram, get_peak_ram())
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses, peak_ram

def evaluate(model, x, y):
    with torch.no_grad():
        outputs = model(x)
        predicted = (outputs > 0.5).float()
        acc = (predicted == y).float().mean().item()
        return acc, outputs

# --- Experiment Execution ---

def run_experiments():
    torch.manual_seed(42)
    x = torch.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

    # 1. Experiment 1: XOR with Gemini Rule
    print("Running Experiment 1: XOR with Gemini Rule (K=5)...")
    model1 = HWSNetwork(k_harmonics=5)
    losses1, peak_ram1 = train_gemini_rule(model1, x, y)
    acc1, out1 = evaluate(model1, x, y)
    syn_time1, load_time1 = benchmark_weight_gen(model1)

    # 2. Experiment 2: XOR with Backprop
    print("Running Experiment 2: XOR with Backprop (K=5)...")
    model2 = HWSNetwork(k_harmonics=5)
    losses2, peak_ram2 = train_backprop(model2, x, y)
    acc2, out2 = evaluate(model2, x, y)

    # 3. Experiment 3: Scaling K on XOR (using Adam for better expressivity check)
    print("Running Experiment 3: Scaling K (K=1 to 50)...")
    ks = [1, 2, 5, 10, 20, 50]
    scaling_results = []
    for k in ks:
        max_acc = 0
        for seed in [0, 42]:
            torch.manual_seed(seed)
            model_k = HWSNetwork(k_harmonics=k)
            # Use Adam for scaling study as it's better at finding the representation's limit
            train_backprop(model_k, x, y, iterations=5000, lr=0.05, optimizer_type='Adam')
            acc_k, _ = evaluate(model_k, x, y)
            max_acc = max(max_acc, acc_k)
        scaling_results.append((k, max_acc))
        print(f"  K={k}: Max Accuracy = {max_acc*100:.1f}%")

    # Final Reporting
    print("\n" + "="*40)
    print("FINAL EXPERIMENTAL REPORT")
    print("="*40)

    print("\n[Experiment 1: HWS Gemini Rule (K=5)]")
    print(f"Final Accuracy: {acc1*100:.2f}%")
    print(f"Final Loss: {losses1[-1]:.6f}")
    print(f"Peak RAM Usage: {peak_ram1:.2f} MB")
    print(f"Weight Synthesis Time (Ψ): {syn_time1*1e6:.2f} µs")
    print(f"Static Weight Loading Time: {load_time1*1e6:.2f} µs")
    print(f"Coefficient File Size: {model1.alphas.nelement() * model1.alphas.element_size()} bytes")

    print("\n[Experiment 2: HWS Backprop (K=5, SGD)]")
    print(f"Final Accuracy: {acc2*100:.2f}%")
    print(f"Final Loss: {losses2[-1]:.6f}")
    print(f"Peak RAM Usage: {peak_ram2:.2f} MB")

    print("\n[Experiment 3: Scaling K (Adam)]")
    for k, acc in scaling_results:
        print(f"K={k:2d}: Max Accuracy = {acc*100:6.1f}%")

    # Plotting
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses1, label='Gemini Rule')
    plt.plot(losses2, label='Backprop (SGD)')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curves (K=5)')
    plt.legend()

    plt.subplot(1, 2, 2)
    ks_plt, acc_plt = zip(*scaling_results)
    plt.plot(ks_plt, acc_plt, marker='o')
    plt.xlabel('K')
    plt.ylabel('Max Accuracy')
    plt.title('Scaling K vs XOR Accuracy')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('hws_results_summary.png')
    print("\nSummary plot saved as hws_results_summary.png")

if __name__ == "__main__":
    run_experiments()
