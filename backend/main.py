import base64
import io

import albumentations as A  # type: ignore[import-not-found]
import cv2
import numpy as np
import timm  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]
from albumentations.pytorch import ToTensorV2  # type: ignore[import-not-found]
from fastapi import FastAPI, File, UploadFile  # type: ignore[import-not-found]
from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
from image_quality import assess_quality, enhance_image
from PIL import Image  # type: ignore[import-not-found]
from pytorch_grad_cam import GradCAM  # type: ignore[import-not-found]
from pytorch_grad_cam.utils.image import (
    show_cam_on_image,  # type: ignore[import-not-found]
)
from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,  # type: ignore[import-not-found]
)
from tta import tta_predict

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("best_dr_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

TEMPERATURE = 1.0
REVIEW_CONFIDENCE_THRESHOLD = 0.60
REVIEW_UNCERTAINTY_THRESHOLD = 0.15

REFERABLE_PROB_THRESHOLD = 0.35

MEDICAL_INFO = {
    0: {
        "label": "No DR",
        "action": "Routine Eye Checkup",
        "recommendation": "No retinopathy lesions detected. Advise annual diabetic eye screening and regular blood glucose control.",
    },
    1: {
        "label": "Mild DR",
        "action": "Follow-up in 6-12 Months",
        "recommendation": "Mild microaneurysms detected. Maintain tight glycemic control and re-screen in 6 to 12 months.",
    },
    2: {
        "label": "Moderate DR",
        "action": "Referral to Ophthalmologist",
        "recommendation": "Hemorrhages and hard exudates detected. Refer to an ophthalmologist within 4-6 weeks for detailed evaluation.",
    },
    3: {
        "label": "Severe DR",
        "action": "Urgent Specialist Referral",
        "recommendation": "Multiple hemorrhages and cotton wool spots observed. Urgent referral to a retina specialist within 1-2 weeks.",
    },
    4: {
        "label": "Proliferative DR",
        "action": "Immediate Emergency Referral",
        "recommendation": "Neovascularization detected with high risk of vision loss. Immediate referral for anti-VEGF or laser photocoagulation therapy.",
    },
}

transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


def _looks_like_fundus(rgb_img: np.ndarray) -> bool:
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    hue, sat, val = cv2.split(hsv)
    warm_mask = ((hue < 25) | (hue > 160)) & (sat > 40) & (val > 20)
    warm_fraction = warm_mask.mean()
    return warm_fraction > 0.25


@app.post("/predict")
async def predict(file: UploadFile = File(...)):  # noqa: B008
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    rgb_img = np.array(image)

    # --- quality gate ---
    quality = assess_quality(rgb_img)

    if quality["decision"] == "reject":
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "reason": quality["feedback"] or "image quality insufficient",
                "quality_metrics": quality,
            },
        )

    enhancement_applied = quality["decision"] == "borderline"
    if enhancement_applied:
        rgb_img = enhance_image(rgb_img)

    if not _looks_like_fundus(rgb_img):
        return JSONResponse(
            status_code=422,
            content={
                "status": "rejected",
                "reason": "uploaded image does not appear to be a fundus photograph",
                "quality_metrics": quality,
            },
        )
    predicted_class, confidence, class_probs, uncertainty = tta_predict(
        model, rgb_img, DEVICE, temperature=TEMPERATURE
    )

    needs_review = (
        confidence < REVIEW_CONFIDENCE_THRESHOLD
        or uncertainty > REVIEW_UNCERTAINTY_THRESHOLD
    )

    referable_prob = sum(class_probs[2:])  # P(grade 2) + P(grade 3) + P(grade 4)
    is_referable = referable_prob >= REFERABLE_PROB_THRESHOLD or predicted_class >= 2

    input_tensor = transform(image=rgb_img)["image"].unsqueeze(0).to(DEVICE)
    target_layers = [model.conv_head]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(predicted_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]

    normalized_img = np.float32(cv2.resize(rgb_img, (224, 224))) / 255.0
    visualization = show_cam_on_image(normalized_img, grayscale_cam, use_rgb=True)

    res_img = Image.fromarray(visualization)
    buffered = io.BytesIO()
    res_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

    info = MEDICAL_INFO[predicted_class]

    return {
        "status": "graded",
        "diagnosis": info["label"],
        "severity_grade": predicted_class,
        "confidence": round(confidence, 4),  # calibrated once TEMPERATURE is fitted
        "class_probabilities": [round(p, 4) for p in class_probs],  # full distribution
        "uncertainty": round(uncertainty, 4),  # TTA disagreement
        "needs_human_review": needs_review,  # explicit low-trust flag
        "referable_probability": round(
            referable_prob, 4
        ),  # P(grade>=2), not just argmax
        "is_referable": is_referable,  # actual clinical screening decision
        "action_required": info["action"],
        "clinical_recommendation": info["recommendation"],
        "heatmap_base64": img_str,
        "quality_metrics": quality,
        "enhancement_applied": enhancement_applied,
    }
