"""
Sign Language Recognition - Training Script (PyTorch)
Trains CustomCNN or TransferCNN on the ASL alphabet dataset using your RTX 3060.
"""

import os
import json
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_preprocessing import get_dataloaders, DATASET_DIR
from model import build_model

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_TYPE = "transfer"    # "custom" | "transfer"
EPOCHS     = 30
LR         = 1e-3
SAVE_DIR   = "saved_models"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── One epoch of training ────────────────────────────────────────────────────
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─── One epoch of validation ──────────────────────────────────────────────────
@torch.no_grad()
def val_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss    = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct    += (outputs.argmax(1) == labels).sum().item()
        total      += images.size(0)

    return total_loss / total, correct / total


# ─── Plot training curves ─────────────────────────────────────────────────────
def plot_history(history, save_path="training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history["train_acc"], label="Train")
    axes[0].plot(history["val_acc"],   label="Val")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_loss"], label="Train")
    axes[1].plot(history["val_loss"],   label="Val")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved → {save_path}")


# ─── Main training loop ───────────────────────────────────────────────────────
def train(model_type: str = MODEL_TYPE):
    print(f"\n{'='*50}")
    print(f"  Device : {DEVICE}")
    print(f"  Model  : {model_type.upper()} CNN")
    print(f"{'='*50}\n")

    os.makedirs(SAVE_DIR, exist_ok=True)

    train_loader, val_loader, class_names = get_dataloaders(DATASET_DIR)

    model     = build_model(model_type, num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                  patience=3, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 7

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   []}

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:>3}/{EPOCHS} | "
              f"Train Loss {train_loss:.4f}  Acc {train_acc*100:.2f}% | "
              f"Val Loss {val_loss:.4f}  Acc {val_acc*100:.2f}% | "
              f"{elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(SAVE_DIR, f"{model_type}_best.pth"))
            print(f"  ✓ New best val accuracy: {best_val_acc*100:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # Save final model + class map
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{model_type}_final.pth"))
    with open(os.path.join(SAVE_DIR, "class_names.json"), "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nBest Val Accuracy : {best_val_acc*100:.2f}%")
    print(f"Models saved to   : {SAVE_DIR}/")

    plot_history(history)
    return model


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train(model_type=MODEL_TYPE)