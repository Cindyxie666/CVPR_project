import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models, datasets
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
BATCH_SIZE = 64
EPOCHS = 30
LR = 3e-4
VAL_RATIO = 0.1
SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / total, 100.0 * correct / total


def main():
    print(f"设备: {DEVICE}")
    set_seed(SEED)

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    base_train_dataset = datasets.ImageFolder(TRAIN_DIR)
    train_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_dataset_full = datasets.ImageFolder(TRAIN_DIR, transform=test_tf)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_tf)

    LABELS = base_train_dataset.classes
    if test_dataset.classes != LABELS:
        raise ValueError(f"训练/测试类别不一致: {LABELS} vs {test_dataset.classes}")

    num_samples = len(base_train_dataset)
    val_size = max(1, int(num_samples * VAL_RATIO))
    train_size = num_samples - val_size
    if train_size <= 0:
        raise ValueError("验证集划分过大，训练集为空。请减小 VAL_RATIO。")

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(num_samples, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    print(f"类别: {LABELS}")
    print(f"训练: {len(train_dataset)} 张 | 验证: {len(val_dataset)} 张 | 测试: {len(test_dataset)} 张")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, len(LABELS))
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        status = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            status = " BEST"

        print(f"[{epoch + 1}/{EPOCHS}] Loss:{train_loss:.4f} "
              f"TrainAcc:{train_acc:.1f}% ValLoss:{val_loss:.4f} "
              f"ValAcc:{val_acc:.1f}%{status}")

    print(f"\nBest ValAcc: {best_val_acc:.2f}%")

    model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"Final TestLoss: {test_loss:.4f} | Final TestAcc: {test_acc:.2f}%")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=LABELS))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title(f'Confusion Matrix (Test Acc: {test_acc:.1f}%)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    print("confusion_matrix.png saved")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.plot(history['train_loss'], label='Train Loss')
    a1.plot(history['val_loss'], label='Val Loss')
    a1.set_title('Loss')
    a1.set_xlabel('Epoch')
    a1.legend()
    a2.plot(history['train_acc'], label='Train Acc')
    a2.plot(history['val_acc'], label='Val Acc')
    a2.set_title('Accuracy (%)')
    a2.set_xlabel('Epoch')
    a2.legend()
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    plt.show()
    print("training_curves.png saved")
    print("\nDone!")


if __name__ == '__main__':
    main()
