"""
Sign Language Recognition - Model Architectures (PyTorch)
Two options:
  1. CustomCNN    - built from scratch
  2. TransferCNN  - MobileNetV2 backbone (higher accuracy)
"""

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 26


# ─── 1. Custom CNN ────────────────────────────────────────────────────────────
class CustomCNN(nn.Module):
    """
    3-block CNN with BatchNorm and Dropout.
    ~2M parameters - fast to train, good baseline.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        def conv_block(in_ch, out_ch, dropout=0.25):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            )

        self.features = nn.Sequential(
            conv_block(3,   32,  dropout=0.25),   # 64 → 32
            conv_block(32,  64,  dropout=0.25),   # 32 → 16
            conv_block(64,  128, dropout=0.40),   # 16 → 8
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ─── 2. Transfer CNN (MobileNetV2) ───────────────────────────────────────────
class TransferCNN(nn.Module):
    """
    MobileNetV2 pre-trained on ImageNet.
    Fine-tunes the last few layers for ASL classification.
    Targets 97%+ accuracy.
    """
    def __init__(self, num_classes=NUM_CLASSES, fine_tune_layers=20):
        super().__init__()

        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # Freeze all layers first
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze last `fine_tune_layers` feature layers
        features = list(backbone.features.children())
        for layer in features[-fine_tune_layers:]:
            for param in layer.parameters():
                param.requires_grad = True

        self.features = backbone.features
        self.pool     = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.last_channel   # 1280 for MobileNetV2
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.50),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ─── Factory function ─────────────────────────────────────────────────────────
def build_model(model_type: str = "custom", num_classes: int = NUM_CLASSES):
    if model_type == "transfer":
        return TransferCNN(num_classes=num_classes)
    return CustomCNN(num_classes=num_classes)


# ─── Sanity check ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dummy = torch.randn(2, 3, 64, 64)

    m1 = build_model("custom")
    print("CustomCNN output :", m1(dummy).shape)
    total = sum(p.numel() for p in m1.parameters())
    print(f"Parameters       : {total:,}")

    m2 = build_model("transfer")
    print("\nTransferCNN output:", m2(dummy).shape)
    trainable = sum(p.numel() for p in m2.parameters() if p.requires_grad)
    print(f"Trainable params  : {trainable:,}")