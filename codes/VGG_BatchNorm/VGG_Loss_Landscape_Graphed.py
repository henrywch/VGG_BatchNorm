import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader


# Classification accuracy calculation
def get_accuracy(model, loader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


# Seed setting for reproducibility
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# Modified training function with per-batch loss tracking
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    all_batch_losses = []  # Track losses for every batch

    batches_n = len(train_loader)
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad = []

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            grad_norm = model.features[0].weight.grad.norm().item()
            grad.append(grad_norm)

            optimizer.step()

            loss_value = loss.item()
            loss_list.append(loss_value)
            all_batch_losses.append(loss_value)  # Per-batch recording

        grads.append(grad)
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc
        learning_curve[epoch] = np.mean(loss_list)

        # Progress plotting
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))
        axes[0].plot(learning_curve)
        axes[1].plot(train_accuracy_curve[:epoch + 1], label='Train')
        axes[1].plot(val_accuracy_curve[:epoch + 1], label='Val')
        plt.savefig(os.path.join(figures_path, f'progress_epoch_{epoch}.png'))
        plt.close()

    return all_batch_losses, grads  # Return per-batch losses


# Integrated plotting function with both curves
def plot_loss_landscape(bn_losses, nobn_losses):
    plt.figure(figsize=(10, 6))

    # Calculate steps and curves
    steps = np.arange(len(bn_losses[0]))

    # BatchNorm results
    bn_min = np.min(bn_losses, axis=0)
    bn_max = np.max(bn_losses, axis=0)
    bn_mean = np.mean(bn_losses, axis=0)

    # No BatchNorm results
    nobn_min = np.min(nobn_losses, axis=0)
    nobn_max = np.max(nobn_losses, axis=0)
    nobn_mean = np.mean(nobn_losses, axis=0)

    # Plot both curves
    plt.plot(steps, bn_mean, color='blue', label='With BatchNorm')
    plt.fill_between(steps, bn_min, bn_max, color='blue', alpha=0.1)

    plt.plot(steps, nobn_mean, color='red', label='Without BatchNorm')
    plt.fill_between(steps, nobn_min, nobn_max, color='red', alpha=0.1)

    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss Landscape Comparison')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'combined_loss_landscape.png'))
    plt.close()


if __name__ == "__main__":
    # Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    epochs = 20
    learning_rates = [1e-4, 5e-4, 1e-3, 2e-3]

    # Path setup
    module_path = os.path.dirname(os.getcwd())
    figures_path = os.path.join(module_path, 'reports', 'figures')
    os.makedirs(figures_path, exist_ok=True)

    # Data loading
    train_loader = get_cifar_loader(train=True, batch_size=batch_size)
    val_loader = get_cifar_loader(train=False, batch_size=batch_size)

    # Training loop
    all_losses = {'with_bn': [], 'without_bn': []}

    for lr in learning_rates:
        # With BN
        set_random_seeds(2020, device)
        model = VGG_A_BatchNorm().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * (batch_size / 128))
        losses, _ = train(model, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs_n=epochs)
        all_losses['with_bn'].append(losses)

        # Without BN
        set_random_seeds(2020, device)
        model = VGG_A().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr * (batch_size / 128))
        losses, _ = train(model, optimizer, nn.CrossEntropyLoss(), train_loader, val_loader, epochs_n=epochs)
        all_losses['without_bn'].append(losses)

    # Align sequence lengths
    min_length = min(
        min(len(seq) for seq in all_losses['with_bn']),
        min(len(seq) for seq in all_losses['without_bn'])
    )

    bn_losses = np.array([seq[:min_length] for seq in all_losses['with_bn']])
    nobn_losses = np.array([seq[:min_length] for seq in all_losses['without_bn']])

    # Generate final plot
    plot_loss_landscape(bn_losses, nobn_losses)