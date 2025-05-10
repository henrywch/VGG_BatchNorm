import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

def main():
    # Hyperparameters
    batch_size = 128
    epochs = 200
    lr = 0.001
    weight_decay = 0.05
    smoothing = 0.1
    optimizer_type = "AdamW"  # Options: "AdamW" or "SGD"
    activation_type = "Swish"  # Options: "Swish", "GELU", "Mish"

    # ----------------------------
    # 1. Data Loading and Augmentation
    # ----------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),  # RandAugment for better generalization
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----------------------------
    # 2. Enhanced ResNet with Swish/GELU/Mish Activation
    # ----------------------------
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    class Mish(nn.Module):
        def forward(self, x):
            return x * torch.tanh(torch.nn.functional.softplus(x))

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super(BasicBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != self.expansion * planes:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

            # Select activation
            if activation_type == "Swish":
                self.activation = Swish()
            elif activation_type == "GELU":
                self.activation = nn.GELU()
            elif activation_type == "Mish":
                self.activation = Mish()
            else:
                self.activation = nn.ReLU()

        def forward(self, x):
            out = self.activation(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = self.activation(out)
            return out

    class ResNet(nn.Module):
        def __init__(self, block, num_blocks, num_classes=10):
            super(ResNet, self).__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
            self.linear = nn.Linear(256 * block.expansion, num_classes)
            self.activation = Swish() if activation_type == "Swish" else nn.GELU()

        def _make_layer(self, block, planes, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.in_planes, planes, stride))
                self.in_planes = planes * block.expansion
            return nn.Sequential(*layers)

        def forward(self, x):
            out = self.activation(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = nn.AdaptiveAvgPool2d((1, 1))(out)
            out = torch.flatten(out, 1)
            out = self.linear(out)
            return out

    def ResNet18():
        return ResNet(BasicBlock, [2, 2, 2, 2])

    # ----------------------------
    # 3. Label Smoothing Cross-Entropy Loss
    # ----------------------------
    class LabelSmoothingCrossEntropy(nn.Module):
        def __init__(self, smoothing=0.1, reduction='mean'):
            super(LabelSmoothingCrossEntropy, self).__init__()
            self.smoothing = smoothing
            self.reduction = reduction

        def forward(self, preds, target):
            n = preds.size()[-1]
            log_preds = torch.log_softmax(preds, dim=-1)
            loss = -log_preds.gather(dim=-1, index=target.unsqueeze(1))
            loss = ((1 - self.smoothing) * loss) - (self.smoothing / n) * log_preds.sum(dim=-1)
            return loss.mean() if self.reduction == 'mean' else loss.sum()

    # ----------------------------
    # 4. Model, Optimizer, and Scheduler
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18().to(device)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    # Select optimizer
    if optimizer_type == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError("Unsupported optimizer type")

    # Cosine Annealing with Warm Restarts for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # ----------------------------
    # 5. Training Loop with Metrics Tracking
    # ----------------------------
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    lrs = []

    def train(epoch):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
        for i, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Metrics
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Record learning rate
            if i % 100 == 99:
                lrs.append(optimizer.param_groups[0]['lr'])
                print(f"Epoch {epoch}, Batch {i+1}, Loss: {running_loss/100:.3f}")
                running_loss = 0.0

        scheduler.step()
        train_losses.append(loss.item())
        train_accuracies.append(100 * correct / total)

    def test(epoch):
        model.eval()
        correct, total = 0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = criterion(model(inputs), targets).item()
        val_losses.append(val_loss)
        val_accuracies.append(100 * correct / total)
        print(f"Epoch {epoch} | Test Accuracy: {100 * correct / total:.2f}% | Loss: {val_loss:.4f}")
        return all_preds, all_targets

    # ----------------------------
    # 6. Training and Evaluation
    # ----------------------------
    for epoch in range(1, epochs + 1):
        train(epoch)
        if epoch % 5 == 0:  # Reduce logging frequency
            preds, targets = test(epoch)

    # ----------------------------
    # 7. Visualization
    # ----------------------------
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.show()

    # Confusion Matrix
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=trainset.classes, yticklabels=trainset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

if __name__ == "__main__":
    main()