import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from sklearn.metrics import confusion_matrix, classification_report


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 2


def get_test_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Dropout(0.6),
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, num_classes),
    )
    return model.to(DEVICE)


class PairedExpressionDataset(Dataset):
    def __init__(self, orig_dir: str, anonymised_dir: str, transform: transforms.Compose):
        self.orig_dir = orig_dir
        self.anonymised_dir = anonymised_dir
        self.transform = transform

        orig_dataset = datasets.ImageFolder(orig_dir)
        anon_dataset = datasets.ImageFolder(anonymised_dir)

        if orig_dataset.classes != anon_dataset.classes:
            raise ValueError(
                f"orig/anon 类别不一致: {orig_dataset.classes} vs {anon_dataset.classes}"
            )

        self.classes = orig_dataset.classes
        orig_map = self._build_sample_map(orig_dataset.samples, orig_dir)
        anon_map = self._build_sample_map(anon_dataset.samples, anonymised_dir)

        orig_keys = set(orig_map)
        anon_keys = set(anon_map)
        if orig_keys != anon_keys:
            missing_in_anon = sorted(orig_keys - anon_keys)[:5]
            missing_in_orig = sorted(anon_keys - orig_keys)[:5]
            raise ValueError(
                "orig/anon 文件不完全对应。"
                f" missing_in_anon={missing_in_anon}, missing_in_orig={missing_in_orig}"
            )

        self.samples = []
        for rel_path in sorted(orig_map):
            orig_path, orig_label = orig_map[rel_path]
            anon_path, anon_label = anon_map[rel_path]
            if orig_label != anon_label:
                raise ValueError(
                    f"标签不一致: {rel_path} -> orig={orig_label}, anon={anon_label}"
                )
            self.samples.append((orig_path, anon_path, orig_label, rel_path))

    @staticmethod
    def _build_sample_map(samples: list[tuple[str, int]], root_dir: str) -> dict[str, tuple[str, int]]:
        return {
            os.path.relpath(path, root_dir): (path, label)
            for path, label in samples
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        orig_path, anon_path, label, _rel_path = self.samples[idx]
        orig_img = Image.open(orig_path).convert("RGB")
        anon_img = Image.open(anon_path).convert("RGB")
        return self.transform(orig_img), self.transform(anon_img), label


def get_predictions(
    model: nn.Module, loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    orig_preds: List[int] = []
    anon_preds: List[int] = []
    labels_all: List[int] = []
    with torch.no_grad():
        for orig_imgs, anon_imgs, labels in loader:
            orig_imgs = orig_imgs.to(DEVICE)
            anon_imgs = anon_imgs.to(DEVICE)

            orig_out = model(orig_imgs)
            anon_out = model(anon_imgs)

            orig_preds.extend(orig_out.argmax(1).cpu().numpy())
            anon_preds.extend(anon_out.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    return (
        np.array(orig_preds, dtype=np.int64),
        np.array(anon_preds, dtype=np.int64),
        np.array(labels_all, dtype=np.int64),
    )


def main(
    orig_dir: str = "data/test",
    anonymised_dir: str = "data/anonymised_test",
    weights_path: str = "best_model.pth",
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> None:
    print(f"设备: {DEVICE}")
    print(f"原始图像目录: {orig_dir}")
    print(f"匿名图像目录: {anonymised_dir}")
    print("评估方式: 使用 hold-out test set，仅做最终测试，不参与模型选择。")

    tf = get_test_transform()
    paired_dataset = PairedExpressionDataset(orig_dir, anonymised_dir, tf)
    labels = paired_dataset.classes
    print(f"类别: {labels}")
    print(f"样本数: {len(paired_dataset)}")

    loader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(num_classes=len(labels))
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"未找到权重文件 {weights_path}，请先在 train.py 中完成训练并保存 best_model.pth。"
        )
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))

    print("对测试集中的原始/匿名图像进行表情预测...")
    orig_preds, anon_preds, all_labels = get_predictions(model, loader)

    orig_acc = (orig_preds == all_labels).mean() * 100.0
    anon_acc = (anon_preds == all_labels).mean() * 100.0
    consistency = (orig_preds == anon_preds).mean() * 100.0
    print(f"\nOriginal Test Accuracy   : {orig_acc:.2f}%")
    print(f"Anonymised Test Accuracy : {anon_acc:.2f}%")
    print(f"Accuracy Drop            : {orig_acc - anon_acc:.2f}%")
    print(f"Expression Consistency   : {consistency:.2f}%")

    print("\n匿名图像分类报告:")
    print(classification_report(all_labels, anon_preds, target_names=labels))

    cm = confusion_matrix(orig_preds, anon_preds)
    print("\n原图预测 vs 匿名图预测 混淆矩阵（行：原图预测，列：匿名图预测）:")
    print(cm)
    print(f"类别顺序: {labels}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate expression consistency on the held-out test set",
    )
    parser.add_argument("--orig_dir", type=str, default="data/test")
    parser.add_argument("--anonymised_dir", type=str, default="data/anonymised_test")
    parser.add_argument("--weights_path", type=str, default="best_model.pth")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    args = parser.parse_args()
    main(
        orig_dir=args.orig_dir,
        anonymised_dir=args.anonymised_dir,
        weights_path=args.weights_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

