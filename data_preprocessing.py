"""
Sign Language Recognition - Data Preprocessing & Augmentation (PyTorch)
Dataset: ASL Alphabet from Kaggle
https://www.kaggle.com/datasets/grassknoted/asl-alphabet
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE    = 64
BATCH_SIZE  = 32
NUM_CLASSES = 29
DATASET_DIR = "archive/asl_alphabet_train/asl_alphabet_train"
SEED        = 42


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        # Only add noise to the tensor
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
    

# ─── Transforms ───────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), shear=10, scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    # Add noise here: std=0.05 is a good starting point (5% noise)
    AddGaussianNoise(0., 0.05), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)), # Zoom in/out randomly
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Change hand angle
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ─── Build DataLoaders ────────────────────────────────────────────────────────
def get_dataloaders(dataset_dir: str = DATASET_DIR):
    """
    Returns (train_loader, val_loader, class_names).
    Deterministic 80/20 split with no index overlap between train and val.
    """
    # Generate split indices once using a fixed seed
    full_dataset = datasets.ImageFolder(dataset_dir, transform=val_transforms)
    class_names  = full_dataset.classes
    n            = len(full_dataset)

    generator     = torch.Generator().manual_seed(SEED)
    indices       = torch.randperm(n, generator=generator).tolist()
    val_size      = int(0.20 * n)
    train_indices = indices[val_size:]   # first 80%
    val_indices   = indices[:val_size]   # last 20%

    # Two ImageFolder instances so each split gets its own transform
    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(dataset_dir, transform=val_transforms)

    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset,   val_indices)

    # Sanity check - no overlap
    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} overlapping indices!"
    print(f"  Train samples : {len(train_indices)}")
    print(f"  Val samples   : {len(val_indices)}")
    print(f"  Overlap       : {len(overlap)}  ✓")

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names


# ─── Visualise a batch ────────────────────────────────────────────────────────
def visualise_samples(loader, class_names, n=16):
    images, labels = next(iter(loader))
    images = images * 0.5 + 0.5
    images = images.permute(0, 2, 3, 1).numpy()

    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2.5))
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        ax.imshow(np.clip(images[i], 0, 1))
        ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis("off")
    plt.suptitle("Sample ASL Alphabet Images", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("Saved -> sample_images.png")


# ─── Dataset stats ────────────────────────────────────────────────────────────
def dataset_stats(train_loader, val_loader, class_names):
    print(f"\n{'─'*40}")
    print(f"  Classes        : {len(class_names)}")
    print(f"  Train samples  : {len(train_loader.dataset)}")
    print(f"  Val samples    : {len(val_loader.dataset)}")
    print(f"  Train batches  : {len(train_loader)}")
    print(f"  Image size     : {IMG_SIZE}x{IMG_SIZE}")
    print(f"{'─'*40}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.isdir(DATASET_DIR):
        print(
            f"[!] Dataset folder '{DATASET_DIR}' not found.\n"
            "    Download: https://www.kaggle.com/datasets/grassknoted/asl-alphabet\n"
            "    Unzip so A-Z folders sit inside archive/asl_alphabet_train/"
        )
    else:
        train_loader, val_loader, class_names = get_dataloaders()
        dataset_stats(train_loader, val_loader, class_names)
        visualise_samples(train_loader, class_names)