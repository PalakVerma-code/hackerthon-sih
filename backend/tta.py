"""
tta.py -- Test-time augmentation for more robust inference on noisy field images.

Single forward pass is fragile to the exact crop/brightness/orientation of a
portable-camera capture. Averaging predictions over a few cheap augmented
views reduces variance without retraining anything.
"""

import albumentations as A  # type: ignore[import-not-found]
import numpy as np
import torch  # type: ignore
from albumentations.pytorch import ToTensorV2  # type: ignore[import-not-found]

BASE = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

TTA_TRANSFORMS = [
    BASE,
    A.Compose(
        [
            A.Resize(224, 224),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    A.Compose(
        [
            A.Resize(224, 224),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
    A.Compose(
        [
            A.Resize(224, 224),
            A.Rotate(limit=10, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    ),
]


def tta_predict(model, rgb_img: np.ndarray, device, temperature: float = 1.0):
    """
    Runs N augmented views through the model, averages softmax probabilities.
    Returns (predicted_class, mean_confidence, per_class_probs, prediction_std)
    prediction_std is a cheap uncertainty signal: high std across views = flag for review.
    """
    all_probs = []
    with torch.no_grad():
        for t in TTA_TRANSFORMS:
            tensor = t(image=rgb_img)["image"].unsqueeze(0).to(device)
            logits = model(tensor)
            probs = torch.softmax(logits / temperature, dim=1)
            all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    mean_probs = all_probs.mean(axis=0)
    pred_class = int(mean_probs.argmax())
    confidence = float(mean_probs[pred_class])
    uncertainty = float(all_probs[:, pred_class].std())

    return pred_class, confidence, mean_probs.tolist(), uncertainty
