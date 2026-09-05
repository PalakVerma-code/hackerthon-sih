import albumentations as A
import timm
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import nn

# ---------------------------------------------------------------------------
# 1. Domain-realistic augmentation
# ---------------------------------------------------------------------------
train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.OneOf(
            [
                A.ImageCompression(quality_lower=40, quality_upper=90, p=1.0),
                A.GaussNoise(var_limit=(10, 50), p=1.0),
            ],
            p=0.3,
        ),
        A.RandomShadow(p=0.15),
        A.CoarseDropout(max_holes=3, max_height=20, max_width=20, p=0.15),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


# ---------------------------------------------------------------------------
# 2. Class-weighted focal loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Down-weights easy (majority-class) examples, focuses learning on the
    hard/rare classes (Severe, Proliferative DR)."""

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, weight=self.class_weights, reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def compute_class_weights(class_counts: list) -> torch.Tensor:
    """class_counts = [count_grade0, count_grade1, ..., count_grade4] from YOUR training set."""
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = counts.sum() / (len(counts) * counts)
    return weights


# ---------------------------------------------------------------------------
# 3. Ordinal-aware loss (CORAL-style alternative to plain cross-entropy)
# ---------------------------------------------------------------------------
class OrdinalLoss(nn.Module):
    def forward(self, logits, targets):
        num_thresholds = logits.shape[1]
        levels = torch.arange(num_thresholds, device=logits.device).unsqueeze(0)
        binary_targets = (targets.unsqueeze(1) > levels).float()
        return F.binary_cross_entropy_with_logits(logits, binary_targets)


model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=5)
