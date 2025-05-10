import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

# Set matplotlib backend to avoid GUI issues
matplotlib.use('Agg')


# Custom Loss Functions
class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classes"""

    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy"""

    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Custom Optimizer Implementation
class CustomSignSGD(optim.Optimizer):
    """Custom optimizer combining sign-based updates with momentum"""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # Apply weight decay
                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)
                # Update momentum buffer
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.data)
                # Update rule: sign(grad) + momentum
                state['momentum_buffer'].mul_(momentum).add_(torch.sign(grad), alpha=1 - momentum)
                # Apply update
                p.data.add_(state['momentum_buffer'], alpha=-lr)
        return loss


# Data Transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


# CNN Model
class CIFAR10CNN(nn.Module):
    def __init__(self, filters=(32, 64, 128), activation='relu', use_batchnorm=True, dropout_rate=0.5):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, filters[0], 3, padding=1),
            nn.BatchNorm2d(filters[0]) if use_batchnorm else nn.Identity(),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters[0], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]) if use_batchnorm else nn.Identity(),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(filters[1], filters[2], 3, padding=1),
            nn.BatchNorm2d(filters[2]) if use_batchnorm else nn.Identity(),
            self._get_activation(activation),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(filters[2] * 4 * 4, 512),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            self._get_activation(activation),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 10)
        )

    def _get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.05)
        elif activation == 'elu':
            return nn.ELU()
        else:
            raise ValueError("Unsupported activation")

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


# Configuration
model_config = {
    'filters': (128, 256, 512),
    'activation': 'leaky_relu',
    'use_batchnorm': True,
    'dropout_rate': 0.4,
}

training_config = {
    'optimizer': 'swa',  # Now using SWA
    'lr': 0.002,
    'weight_decay': 1e-5,  # L2 regularization
    'momentum': 0.9,
    'l1_lambda': 0.0,  # L1 regularization strength
    'grad_clip': 5.0,  # Gradient clipping
    'epochs': 100,
    'loss': 'label_smoothing',  # Options: cross_entropy, focal, mse, label_smoothing
    'label_smoothing': 0.15,  # For label smoothing
    'focal_params': {'alpha': 0.25, 'gamma': 2},
    'scheduler': {
        'name': 'cosine',
        'step_size': 20,
        'gamma': 0.1,
        'patience': 3,
        'factor': 0.2,
        'min_lr': 1e-6
    },
    # SWA parameters
    'swa_start': 80,  # Start SWA after this epoch
    'swa_lr': 0.05,  # Learning rate during SWA
    'swa_anneal_epochs': 10,  # Annealing period
}

if __name__ == '__main__':
    # Dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./data_cnn', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2, persistent_workers=True)
    testset = torchvision.datasets.CIFAR10(
        root='./data_cnn', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2, persistent_workers=True)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CIFAR10CNN(**model_config).to(device)

    # Loss function selection
    if training_config['loss'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif training_config['loss'] == 'focal':
        criterion = FocalLoss(**training_config['focal_params'])
    elif training_config['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif training_config['loss'] == 'label_smoothing':
        criterion = LabelSmoothingCrossEntropy(training_config['label_smoothing'])
    else:
        raise ValueError("Invalid loss function")

    # Optimizer
    if training_config['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=training_config['lr'],
                               weight_decay=training_config['weight_decay'])
    elif training_config['optimizer'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=training_config['lr'], momentum=0.9,
                              weight_decay=training_config['weight_decay'])
    elif training_config['optimizer'] == 'custom':
        optimizer = CustomSignSGD(model.parameters(),
                                  lr=training_config['lr'],
                                  momentum=training_config['momentum'],
                                  weight_decay=training_config['weight_decay'])
    elif training_config['optimizer'] == 'swa':
        # Create base optimizer (SGD with momentum)
        base_optimizer = optim.SGD(model.parameters(), lr=training_config['lr'],
                                   momentum=training_config['momentum'],
                                   weight_decay=training_config['weight_decay'])
        # Create SWA model and scheduler
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer=base_optimizer,
                              anneal_strategy='linear',
                              anneal_epochs=training_config['swa_anneal_epochs'],
                              swa_lr=training_config['swa_lr'])
        # Assign base_optimizer as the optimizer
        optimizer = base_optimizer

    # Scheduler
    scheduler_config = training_config['scheduler']
    scheduler = None
    if scheduler_config['name'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'],
                                              gamma=scheduler_config['gamma'])
    elif scheduler_config['name'] == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', threshold=0.001, threshold_mode='rel',
                                                         cooldown=2,
                                                         patience=scheduler_config['patience'],
                                                         factor=scheduler_config['factor'])
    elif scheduler_config['name'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=training_config['epochs'],
                                                         eta_min=scheduler_config['min_lr'])

    # Training loop
    best_val_acc = 0
    train_loss, val_loss = [], []
    train_acc, val_acc = [], []

    for epoch in range(training_config['epochs']):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Convert targets for MSE
            if training_config['loss'] == 'mse':
                targets = F.one_hot(labels, num_classes=10).float().to(device)
            else:
                targets = labels

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # L1 regularization
            if training_config['l1_lambda'] > 0:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss += training_config['l1_lambda'] * l1_loss

            loss.backward()

            # Gradient clipping
            if training_config['grad_clip'] > 0:
                nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip'])

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Validation
        model.eval()
        val_epoch_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                # Convert validation targets for MSE
                if training_config['loss'] == 'mse':
                    val_targets = F.one_hot(labels, num_classes=10).float().to(device)
                else:
                    val_targets = labels

                outputs = model(inputs)
                loss = criterion(outputs, val_targets)

                val_epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        # Update LR
        if training_config['optimizer'] == 'swa' and epoch >= training_config['swa_start']:
            # Update SWA model
            swa_model.update_parameters(model)
            # Step SWA scheduler
            swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_epoch_loss / len(testloader))
                else:
                    scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

        # Save metrics
        train_loss.append(epoch_loss / len(trainloader))
        train_acc.append(100 * correct / total)
        val_loss.append(val_epoch_loss / len(testloader))
        val_acc.append(100 * val_correct / val_total)

        # Save best model
        if val_acc[-1] > best_val_acc:
            best_val_acc = val_acc[-1]
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1}/{training_config["epochs"]}: LR: {current_lr:.2e} | '
              f'Train Loss: {train_loss[-1]:.4f}, Acc: {train_acc[-1]:.2f}% | '
              f'Val Loss: {val_loss[-1]:.4f}, Acc: {val_acc[-1]:.2f}%')

    # Final evaluation
    if training_config['optimizer'] == 'swa':
        # Update BN layers for SWA model
        update_bn(trainloader, swa_model, device=device)
        # Use SWA model for final evaluation
        model = swa_model
        # Save SWA model
        torch.save(model.state_dict(), 'best_model_swa.pth')
    else:
        model.load_state_dict(torch.load('best_model.pth'))

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')

    # Plot results
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Train')
    plt.plot(val_loss, label='Validation')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label='Train')
    plt.plot(val_acc, label='Validation')
    plt.title('Accuracy Curve')
    plt.legend()
    os.makedirs("images", exist_ok=True)
    plt.savefig('images/training_metrics.png')
    plt.close()