"""
Sign Language Recognition - Real-time Webcam Inference (PyTorch)
Press Q to quit.
"""

import json
import os
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

from model import build_model

SAVE_DIR   = "saved_models"
MODEL_TYPE = "transfer"    # "custom" | "transfer"
IMG_SIZE   = 64
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROI_TOP, ROI_LEFT     = 100, 400
ROI_BOTTOM, ROI_RIGHT = 400, 700
CONF_THRESHOLD        = 0.70

# Inference transform (no augmentation)
infer_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# ─── Load model ───────────────────────────────────────────────────────────────
def load_model():
    with open(os.path.join(SAVE_DIR, "class_names.json")) as f:
        class_names = json.load(f)

    model = build_model(MODEL_TYPE, num_classes=len(class_names))
    model.load_state_dict(
        torch.load(os.path.join(SAVE_DIR, f"{MODEL_TYPE}_best.pth"),
                   map_location=DEVICE)
    )
    model.to(DEVICE).eval()
    return model, class_names


# ─── Preprocess ROI ───────────────────────────────────────────────────────────
def preprocess(roi_bgr: np.ndarray) -> torch.Tensor:
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(roi_rgb)
    tensor  = infer_transform(pil_img).unsqueeze(0)   # (1, 3, H, W)
    return tensor.to(DEVICE)


# ─── Main loop ────────────────────────────────────────────────────────────────
def run_webcam():
    model, class_names = load_model()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[!] Cannot access webcam.")
        return

    print(f"Running on: {DEVICE}")
    print("Webcam started — press Q to quit.")
    print(f"Hold your hand inside the green box.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)   # mirror

            # Draw ROI
            cv2.rectangle(frame, (ROI_LEFT, ROI_TOP),
                          (ROI_RIGHT, ROI_BOTTOM), (0, 255, 0), 2)
            cv2.putText(frame, "Place hand here", (ROI_LEFT, ROI_TOP - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            roi    = frame[ROI_TOP:ROI_BOTTOM, ROI_LEFT:ROI_RIGHT]
            tensor = preprocess(roi)

            probs  = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            conf   = float(probs.max())
            letter = class_names[int(probs.argmax())]

            # Main prediction label
            if conf >= CONF_THRESHOLD:
                label  = f"{letter}  {conf*100:.1f}%"
                colour = (0, 230, 0)
            else:
                label  = f"?  ({conf*100:.1f}%)"
                colour = (0, 60, 255)

            cv2.putText(frame, label, (ROI_LEFT, ROI_TOP - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, colour, 3)

            # Top-3 sidebar
            top3 = probs.argsort()[::-1][:3]
            for rank, idx in enumerate(top3):
                text = f"{class_names[idx]}: {probs[idx]*100:.1f}%"
                cv2.putText(frame, text, (10, 40 + rank * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("ASL Sign Language Recognition  (Q = quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()