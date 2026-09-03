from fastapi import FastAPI, UploadFile, File
import torch, cv2, io, base64, timm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

app = FastAPI()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("best_dr_model.pth", map_location=DEVICE))
model.eval().to(DEVICE)

# Medical Diagnosis Details & Clinical Recommendations
MEDICAL_INFO = {
    0: {
        "label": "No DR",
        "action": "Routine Eye Checkup",
        "recommendation": "No retinopathy lesions detected. Advise annual diabetic eye screening and regular blood glucose control."
    },
    1: {
        "label": "Mild DR",
        "action": "Follow-up in 6-12 Months",
        "recommendation": "Mild microaneurysms detected. Maintain tight glycemic control and re-screen in 6 to 12 months."
    },
    2: {
        "label": "Moderate DR",
        "action": "Referral to Ophthalmologist",
        "recommendation": "Hemorrhages and hard exudates detected. Refer to an ophthalmologist within 4-6 weeks for detailed evaluation."
    },
    3: {
        "label": "Severe DR",
        "action": "Urgent Specialist Referral",
        "recommendation": "Multiple hemorrhages and cotton wool spots observed. Urgent referral to a retina specialist within 1-2 weeks."
    },
    4: {
        "label": "Proliferative DR",
        "action": "Immediate Emergency Referral",
        "recommendation": "Neovascularization detected with high risk of vision loss. Immediate referral for anti-VEGF or laser photocoagulation therapy."
    }
}

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    rgb_img = np.array(image)
    
    input_tensor = transform(image=rgb_img)['image'].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        predicted_class = torch.argmax(outputs, dim=1).item()
        
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
        "diagnosis": info["label"],
        "severity_grade": predicted_class,
        "action_required": info["action"],
        "clinical_recommendation": info["recommendation"],
        "heatmap_base64": img_str
    }