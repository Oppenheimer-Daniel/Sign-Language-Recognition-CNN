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
IMG_SIZE    = 64 # image size for training (64x64 is a good balance of speed and detail)
BATCH_SIZE  = 32 # batch size for training (adjust based on your GPU memory)
NUM_CLASSES = 29 # 26 letters + space + delete + nothing
DATASET_DIR = "archive/asl_alphabet_train/asl_alphabet_train" # path to the unzipped training dataset
if not os.path.exists(DATASET_DIR):
    print(f"Error: Dataset not found at {DATASET_DIR}. Please download it from Kaggle.")
    exit()
SEED        = 42 # fixed seed for reproducibility (important for the train/val split)

# Added Gaussian Noise as a custom transform for more realistic augmentation
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
    # 1. Zoom/Crop
    # scale=(0.4, 1.0) means it can zoom in until only 40% of the hand is visible
    # ratio=(0.9, 1.1) keeps the hand from looking too "stretched"
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
    
    # 2. Hand Orientation (The "Left/Right" handedness)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 3. Angle and Position
    transforms.RandomRotation(20), # Increased from 15 for more variety
    transforms.RandomAffine(0, translate=(0.2, 0.2), shear=10), # More translation
    
    # 4. Lighting and Texture
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(), # Convert to tensor before adding noise
    AddGaussianNoise(0., 0.05), # Add Gaussian noise with mean=0 and std=0.05
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Normalize
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), # Just resize for validation - no augmentation
    transforms.ToTensor(), # Convert to tensor before normalization
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # Normalize
])


# ─── Build DataLoaders ────────────────────────────────────────────────────────
def get_dataloaders(dataset_dir: str = DATASET_DIR):
    """
    Returns (train_loader, val_loader, class_names).
    Deterministic 80/20 split with no index overlap between train and val.
    """
    # Generate split indices once using a fixed seed
    full_dataset = datasets.ImageFolder(dataset_dir, transform=val_transforms) # scans dataset folder and automatically assigns labels based on subfolder names
    class_names  = full_dataset.classes
    n            = len(full_dataset)

    # with a new seed for each run, the train/val split will be different but still reproducible if you use the same seed again
    generator     = torch.Generator().manual_seed(SEED) # creates a random generator with a fixed seed for reproducibility
    indices       = torch.randperm(n, generator=generator).tolist()
    val_size      = int(0.20 * n)
    train_indices = indices[val_size:]   # first 80%
    val_indices   = indices[:val_size]   # last 20%

    # Two ImageFolder instances so each split gets its own transform
    train_dataset = datasets.ImageFolder(dataset_dir, transform=train_transforms) # train gets augmentation
    val_dataset   = datasets.ImageFolder(dataset_dir, transform=val_transforms) # val gets only resizing and normalization

    train_subset = Subset(train_dataset, train_indices)
    val_subset   = Subset(val_dataset,   val_indices)

    # Check for overlap
    overlap = set(train_indices) & set(val_indices)
    assert len(overlap) == 0, f"DATA LEAKAGE: {len(overlap)} overlapping indices!"
    print(f"  Train samples : {len(train_indices)}")
    print(f"  Val samples   : {len(val_indices)}")
    print(f"  Overlap       : {len(overlap)}  ✓")

    # Create DataLoaders with shuffling for training and no shuffling for validation
    # A DataLoader is created for each subset to ensure they use the correct transforms and have no index overlap
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, class_names


# ─── Visualise a batch ────────────────────────────────────────────────────────
def visualise_samples(loader, class_names, n=16):
    # Get a batch of images and labels from the loader
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