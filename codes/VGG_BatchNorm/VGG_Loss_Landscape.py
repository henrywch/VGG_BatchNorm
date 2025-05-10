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
from models.vgg import VGG_A_BatchNorm # you need to implement this network
from data.loaders import get_cifar_loader

# This function is used to calculate the accuracy of model classification
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

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []
    grads = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []  # use this to record the loss value of each step
        grad = []  # use this to record the loss gradient of each step
        learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # Forward + backward
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Record gradients BEFORE step
            grad_norm = model.features[0].weight.grad.norm().item()
            grad.append(grad_norm)

            optimizer.step()  # Only one step call

            loss_list.append(loss.item())

        losses_list.append(loss_list)
        grads.append(grad)
        display.clear_output(wait=True)
        f, axes = plt.subplots(1, 2, figsize=(15, 3))

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        learning_curve[epoch] = np.mean(loss_list)
        axes[0].plot(learning_curve)
        axes[1].plot(train_accuracy_curve[:epoch + 1], label='Train')
        axes[1].plot(val_accuracy_curve[:epoch + 1], label='Val')
        axes[1].set_title("Accuracy Curve")
        plt.savefig(os.path.join(figures_path, f'progress_epoch_{epoch}.png'))
        plt.close()

    return learning_curve, grads

# Use this function to plot the final loss landscape,
# fill the area between the two curves can use plt.fill_between()
def plot_loss_landscape():
    plt.figure(figsize=(12, 6))

    # With BN
    plt.subplot(1, 2, 1)
    plt.plot(np.mean(bn_losses, axis=0), label='Mean Loss')
    plt.fill_between(range(epo), min_curve_bn, max_curve_bn, alpha=0.2)
    plt.title("With BatchNorm")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    # Without BN
    plt.subplot(1, 2, 2)
    plt.plot(np.mean(nobn_losses, axis=0), label='Mean Loss')
    plt.fill_between(range(epo), min_curve_nobn, max_curve_nobn, alpha=0.2)
    plt.title("Without BatchNorm")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'loss_landscape.png'))

if __name__ == "__main__":
    # ## Constants (parameters) initialization
    device_id = [0, 1, 2, 3]
    num_workers = 4
    batch_size = 128

    # add our package dir to path
    module_path = os.path.dirname(os.getcwd())
    home_path = module_path
    figures_path = os.path.join(home_path, 'reports', 'figures')
    models_path = os.path.join(home_path, 'reports', 'models')
    os.makedirs(figures_path, exist_ok=True)
    os.makedirs(models_path, exist_ok=True)

    # Make sure you are using the right device.
    device_id = device_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    # Initialize your data loader and
    # make sure that dataloader works
    # as expected by observing one
    # sample from it.
    train_loader = get_cifar_loader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,  # Add this parameter
        shuffle = True
    )
    val_loader = get_cifar_loader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,  # Add this parameter
        shuffle=False
    )
    for X, y in train_loader:
        img = X[0].numpy().transpose((1, 2, 0))
        img = img * 0.5 + 0.5
        plt.imshow(img)
        plt.title(f'Label: {y[0].item()}')
        plt.savefig(os.path.join(figures_path, 'sample_image.png'))
        plt.close()
        break

    # Train your model
    # feel free to modify
    epo = 20
    loss_save_path = 'result/losses'
    grad_save_path = 'result/grads'
    os.makedirs(loss_save_path, exist_ok=True)
    os.makedirs(grad_save_path, exist_ok=True)

    # Train models with different configurations
    all_losses = {'with_bn': {}, 'without_bn': {}}
    # learning_rates = [1e-3, 2e-3, 5e-4]
    # learning_rates = [1e-4]
    # learning_rates = [5e-4]
    learning_rates = [1e-3]

    for lr in learning_rates:
        # Train with BN
        set_random_seeds(2020, device)
        model = VGG_A_BatchNorm()
        optimizer = torch.optim.Adam(model.parameters(), lr=(lr * (batch_size / 128)))
        criterion = nn.CrossEntropyLoss()
        losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
        all_losses['with_bn'][lr] = losses
        np.savetxt(f'{loss_save_path}/bn_lr{lr}.txt', losses)
        np.savetxt(f'{grad_save_path}/grad_bn_lr{lr}.txt', grads)

        # Train without BN
        set_random_seeds(2020, device)
        model = VGG_A()
        optimizer = torch.optim.Adam(model.parameters(), lr=(lr * (batch_size / 128)))
        criterion = nn.CrossEntropyLoss()
        losses, grads = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
        all_losses['without_bn'][lr] = losses
        np.savetxt(f'{loss_save_path}/loss_nobn_lr{lr}.txt', losses)
        np.savetxt(f'{grad_save_path}/grad_nobn_lr{lr}.txt', grads)

    # Maintain two lists: max_curve and min_curve,
    # select the maximum value of loss in all models
    # on the same step, add it to max_curve, and
    # the minimum value to min_curve
    # For models with BN
    bn_losses = np.array([all_losses['with_bn'][lr] for lr in learning_rates])
    min_curve_bn = np.min(bn_losses, axis=0)
    max_curve_bn = np.max(bn_losses, axis=0)

    # For models without BN
    nobn_losses = np.array([all_losses['without_bn'][lr] for lr in learning_rates])
    min_curve_nobn = np.min(nobn_losses, axis=0)
    max_curve_nobn = np.max(nobn_losses, axis=0)

    # Execute plotting
    plot_loss_landscape()