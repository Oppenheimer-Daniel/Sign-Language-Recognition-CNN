"""
Sign Language Recognition - Model Evaluation (PyTorch)
Tests against the held-out asl_alphabet_test folder for honest accuracy.
Generates: classification report, confusion matrix, per-class accuracy chart.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

from model import build_model

# ─── Config ───────────────────────────────────────────────────────────────────
SAVE_DIR   = "saved_models"
MODEL_TYPE = "transfer"      # "custom" | "transfer"
TEST_DIR = "archive/asl_alphabet_test/asl_alphabet_test"
IMG_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ─── Load model ───────────────────────────────────────────────────────────────
def load_model(model_type: str = MODEL_TYPE):
    with open(os.path.join(SAVE_DIR, "class_names.json")) as f:
        class_names = json.load(f)

    model = build_model(model_type, num_classes=len(class_names))
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, f"{model_type}_best.pth"),
                   map_location=DEVICE, weights_only=True)
    )
    model.to(DEVICE).eval()
    print(f"Loaded: {model_type}_best.pth  |  Device: {DEVICE}")
    return model, class_names


# ─── Load test set ────────────────────────────────────────────────────────────
def load_test_loader(class_names: list):
    """
    The asl_alphabet_test folder has one flat folder of images named like
    'A_test.jpg', 'B_test.jpg' etc. We build labels from the filename prefix.
    """
    from PIL import Image
    from torch.utils.data import Dataset

    class ASLTestDataset(Dataset):
        def __init__(self, test_dir, class_names, transform):
            self.transform   = transform
            self.class_names = class_names
            self.samples     = []

            for fname in os.listdir(test_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                # Filename format: "A_test.jpg" → label "A"
                letter = fname.split("_")[0].upper()
                if letter in class_names:
                    label = class_names.index(letter)
                    self.samples.append(
                        (os.path.join(test_dir, fname), label)
                    )

            print(f"  Test images found : {len(self.samples)}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            path, label = self.samples[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label

    dataset = ASLTestDataset(TEST_DIR, class_names, val_transforms)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False,
                         num_workers=2, pin_memory=True)
    return loader


# ─── Collect predictions ──────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader):
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        preds  = model(images).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)


# ─── Confusion matrix ─────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, class_names,
                          save_path="confusion_matrix.png"):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.4, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual",    fontsize=12)
    ax.set_title("Confusion Matrix — Test Set (% per true class)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {save_path}")


# ─── Per-class accuracy ───────────────────────────────────────────────────────
def plot_per_class_accuracy(y_true, y_pred, class_names,
                            save_path="per_class_accuracy.png"):
    cm  = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1) * 100

    colours = ["#d62728" if a < 70 else "#2ca02c" for a in acc]
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(class_names, acc, color=colours, edgecolor="white")
    ax.axhline(acc.mean(), color="navy", ls="--", linewidth=1.5,
               label=f"Mean {acc.mean():.1f}%")
    ax.set_ylim(0, 110)
    ax.set_xlabel("Sign (letter)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-class Recognition Accuracy — Test Set")
    ax.legend()
    ax.bar_label(bars, fmt="%.0f%%", fontsize=7, padding=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved -> {save_path}")


# ─── Full pipeline ────────────────────────────────────────────────────────────
def evaluate(model_type: str = MODEL_TYPE):
    print(f"\n{'='*50}")
    print(f"  Evaluating: {model_type.upper()} CNN on test set")
    print(f"{'='*50}\n")

    model, class_names = load_model(model_type)
    test_loader        = load_test_loader(class_names)

    if len(test_loader.dataset) == 0:
        print("[!] No test images found. Check that TEST_DIR is correct.")
        return

    y_true, y_pred = get_predictions(model, test_loader)

    print("\n── Classification Report ──────────────────────")
    print(classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0))
    print(f"Overall Test Accuracy: {np.mean(y_true == y_pred)*100:.2f}%\n")

    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_per_class_accuracy(y_true, y_pred, class_names)


if __name__ == "__main__":
    evaluate(model_type=MODEL_TYPE)