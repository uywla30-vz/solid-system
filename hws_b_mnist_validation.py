import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import psutil
import os
import matplotlib.pyplot as plt
from hws_b import HWSLinear

def get_ram():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

class HWSMNIST(nn.Module):
    def __init__(self, K=10):
        super(HWSMNIST, self).__init__()
        self.l1 = HWSLinear(784, 128, K=K)
        self.l2 = HWSLinear(128, 10, K=K)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.l1(x))
        x = self.l2(x)
        return x

def run_experiment_6():
    torch.manual_seed(42)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=1000)

    print("--- Experiment 6: HWS-B MNIST Validation ---")
    K_values = [10, 50, 100]
    for K in K_values:
        model = HWSMNIST(K=K)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training a bit more to show potential
        print(f"\nTraining HWS-B with K={K}...")
        for epoch in range(1, 4):
            model.train()
            # Limited batches for speed in this summary script
            for i, (data, target) in enumerate(train_loader):
                if i >= 100: break
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = 0
            with torch.no_grad():
                for i, (data, target) in enumerate(test_loader):
                    if i >= 5: break
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            acc = correct / (5 * 1000)
            print(f"Epoch {epoch}: Test Accuracy {acc*100:.2f}%")

        peak_ram = get_ram()
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        print(f"Result K={K}: Peak RAM={peak_ram:.1f}MB, Param Size={param_size/1024:.2f}KB")

if __name__ == "__main__":
    run_experiment_6()
