# Explainable AI for Diabetic Retinopathy Screening

> **Smart India Hackathon (SIH) Prototype**
> An end-to-end AI-assisted screening system for diabetic retinopathy using retinal fundus images, combining **automated image-quality assessment, robust inference, probability calibration, Grad-CAM explainability, and automated PDF reporting**.

---

## Overview

Diabetic Retinopathy (DR) is a diabetes-related retinal disease that can lead to severe vision impairment if not identified and managed appropriately.

This project presents a modular AI-assisted screening prototype that analyzes retinal fundus images and provides:

* Image-quality assessment before inference
* Five-level DR severity classification
* Test-Time Augmentation (TTA)
* Calibrated prediction probabilities
* Grad-CAM visual explanations
* Screening recommendations
* Automated PDF screening reports

The system follows a **human-in-the-loop approach**, where AI assists screening and prioritization while clinical interpretation remains the responsibility of qualified healthcare professionals.

---

## Medical Disclaimer

This project is a **research, educational, and hackathon prototype**.

It is **not a certified medical device** and must not be used as a substitute for examination, diagnosis, or treatment by a qualified ophthalmologist.

The model predictions, confidence values, recommendations, and Grad-CAM visualizations require appropriate clinical validation before any real-world medical deployment.

---

## Key Features

### 🖼️ Fundus Image Upload

A browser-based Streamlit interface allows users to upload retinal fundus images.

Supported formats:

* PNG
* JPG
* JPEG

Patient metadata can also be entered to generate a structured screening report.

---

### 🔍 Image Quality Assessment

Images are evaluated before model inference to identify potentially unsuitable inputs.

The quality-check pipeline considers factors such as:

* Image blur
* Illumination
* Contrast
* Image artifacts

The objective is to prevent poor-quality images from being blindly passed to the classification model.

```text
Fundus Image
      │
      ▼
Quality Assessment
      │
 ┌────┴────┐
 ▼         ▼
PASS      FAIL
 │         │
 ▼         ▼
AI        Recapture /
Screening Manual Review
```

---

### 🧠 DR Severity Classification

The prototype uses an **EfficientNet-B0** deep-learning architecture for five-class diabetic retinopathy classification.

```text
Fundus Image
      │
      ▼
Preprocessing
      │
      ▼
EfficientNet-B0
      │
      ▼
DR Severity
```

The classification levels are:

| Grade | Severity               |
| ----: | ---------------------- |
| **0** | No DR                  |
| **1** | Mild NPDR              |
| **2** | Moderate NPDR          |
| **3** | Severe NPDR            |
| **4** | Proliferative DR (PDR) |

---

### 🔄 Test-Time Augmentation

The system evaluates multiple transformed versions of the input image during inference.

Possible transformations include:

* Horizontal flipping
* Vertical flipping
* Rotation
* Other inference-time augmentations

The resulting predictions are aggregated to reduce prediction variance and improve robustness against image orientation and difficult edge cases.

```text
                    Fundus Image
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
       Original         Flip         Rotation
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                 Model Predictions
                         │
                         ▼
                    Aggregation
```

---

### 📊 Probability Calibration

Deep-learning models can produce confidence scores that do not always correspond well to actual prediction reliability.

The prototype includes post-hoc probability calibration to improve the interpretation of model confidence.

```text
Raw Model Logits
       │
       ▼
Calibration Layer
       │
       ▼
Calibrated Probabilities
```

This provides more meaningful confidence information for the screening interface.

---

### 🔬 Explainable AI with Grad-CAM

The system generates **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualizations.

Instead of presenting only:

```text
Prediction → Moderate NPDR
```

the system provides:

```text
Prediction
    +
Probability Distribution
    +
Grad-CAM Heatmap
```

The heatmap indicates spatial regions that contributed to the model's prediction.

This can assist with:

* Model interpretation
* Visual inspection
* Debugging
* Research analysis
* Human review

> Grad-CAM indicates model activation patterns and should not be interpreted as a clinically validated lesion detector.

---

### 📄 Automated Screening Reports

The frontend can generate PDF screening reports containing information such as:

* Patient metadata
* Fundus image
* Predicted DR severity
* Probability distribution
* Confidence information
* Screening recommendation
* Grad-CAM visualization

This converts the AI output into a structured, human-readable screening artifact.

---

## System Architecture

The current prototype follows a modular client-server architecture.

```text
┌─────────────────────────────────────┐
│          Streamlit Frontend         │
│                                     │
│ • Patient Metadata                  │
│ • Fundus Image Upload               │
│ • Result Visualization              │
│ • Probability Display               │
│ • Grad-CAM Visualization            │
│ • PDF Report Generation             │
└──────────────────┬──────────────────┘
                   │
                   │ HTTP / Multipart
                   ▼
┌─────────────────────────────────────┐
│           FastAPI Backend           │
│                                     │
│ • API Routing                       │
│ • Image Validation                  │
│ • Image Quality Assessment          │
│ • TTA Inference                     │
│ • Model Prediction                  │
│ • Probability Calibration           │
│ • Grad-CAM Generation               │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          EfficientNet-B0            │
│                                     │
│        DR Severity Prediction       │
└─────────────────────────────────────┘
```

---

## End-to-End Workflow

```text
              Fundus Image
                    │
                    ▼
           Streamlit Interface
                    │
                    ▼
             FastAPI Backend
                    │
                    ▼
          Image Quality Check
                    │
                    ▼
              Preprocessing
                    │
                    ▼
        Test-Time Augmentation
                    │
                    ▼
           EfficientNet-B0
                    │
                    ▼
        Probability Calibration
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
     DR Prediction         Grad-CAM
          │                   │
          └─────────┬─────────┘
                    ▼
            Result Dashboard
                    │
                    ▼
             PDF Screening
                Report
```

---

## Repository Structure

```text
hackerthon-sih/
│
├── backend/
│   ├── best_dr_model.pth
│   ├── calibration.py
│   ├── image_quality.py
│   ├── main.py
│   ├── train_improved.py
│   └── tta.py
│
├── data/
│   └── # Local dataset / validation data
│
├── frontend/
│   ├── app.py
│   ├── temp_heat.jpg
│   └── temp_orig.jpg
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Module Responsibilities

| Module                      | Responsibility                                        |
| --------------------------- | ----------------------------------------------------- |
| `frontend/app.py`           | Streamlit UI, API communication and PDF generation    |
| `backend/main.py`           | FastAPI routing, inference orchestration and Grad-CAM |
| `backend/image_quality.py`  | Image quality assessment                              |
| `backend/tta.py`            | Test-Time Augmentation                                |
| `backend/calibration.py`    | Probability calibration                               |
| `backend/train_improved.py` | Model training and evaluation                         |
| `backend/best_dr_model.pth` | Trained EfficientNet-B0 checkpoint                    |

---

## Technology Stack

| Category            | Technology         | Purpose                                |
| ------------------- | ------------------ | -------------------------------------- |
| Frontend            | Streamlit          | Interactive screening interface        |
| Backend             | FastAPI            | REST API and inference service         |
| Server              | Uvicorn            | ASGI application server                |
| Deep Learning       | PyTorch            | Model training and inference           |
| Model Architecture  | EfficientNet-B0    | DR classification                      |
| Model Library       | `timm`             | EfficientNet implementation            |
| Explainability      | `pytorch-grad-cam` | Grad-CAM generation                    |
| Computer Vision     | OpenCV             | Image-quality analysis                 |
| Image Processing    | Pillow             | Image loading and processing           |
| Augmentation        | Albumentations     | Image transformation and preprocessing |
| Numerical Computing | NumPy              | Numerical operations                   |
| Reporting           | FPDF2              | PDF report generation                  |

---

## Local Setup

### Prerequisites

* Python 3.9+
* Git
* Virtual environment
* CPU or CUDA-compatible GPU

### Clone the Repository

```bash
git clone https://github.com/PalakVerma-code/hackerthon-sih.git

cd hackerthon-sih
```

### Create a Virtual Environment

**Windows PowerShell**

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install --upgrade pip

pip install -r requirements.txt
```

If `fpdf2` is not included in `requirements.txt`:

```bash
pip install fpdf2
```

---

## Running the Application

### Start the FastAPI Backend

Open a terminal, activate the virtual environment, and run:

```bash
cd backend

uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Start the Streamlit Frontend

Open a second terminal, activate the same virtual environment, and run from the project root:

```bash
streamlit run frontend/app.py
```

The Streamlit interface will open in the browser.

---

## API

The primary inference endpoint is:

```http
POST /predict
```

The endpoint accepts the fundus image through a multipart HTTP request along with the required input data.

### Inference Flow

```text
POST /predict
      │
      ▼
Input Validation
      │
      ▼
Image Quality Assessment
      │
      ▼
Preprocessing
      │
      ▼
TTA
      │
      ▼
EfficientNet-B0
      │
      ▼
Probability Calibration
      │
      ▼
Grad-CAM
      │
      ▼
Prediction Response
```

### Example Response

A representative response may look like:

```json
{
    "predicted_class": 2,
    "diagnosis": "Moderate NPDR",
    "confidence": 0.87,
    "probabilities": {
        "No DR": 0.03,
        "Mild NPDR": 0.07,
        "Moderate NPDR": 0.87,
        "Severe NPDR": 0.02,
        "PDR": 0.01
    },
    "quality_status": "PASS"
}
```

> The exact response structure depends on the implementation in `backend/main.py`.

---

## Screening Decision Flow

The prototype is designed around a quality-aware screening process.

```text
                 Upload Image
                      │
                      ▼
             Is Image Suitable?
                 /          \
               No            Yes
               │              │
               ▼              ▼
        Recapture /       AI Inference
        Manual Review          │
                               ▼
                         DR Classification
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              Confidence             Grad-CAM
                    │                     │
                    └──────────┬──────────┘
                               ▼
                       Screening Result
                               │
                               ▼
                         PDF Report
```

---

## Safety & Responsible AI

Medical AI systems require safeguards beyond model accuracy.

### Image Quality Guardrail

Poor-quality images should not automatically be passed to the classifier.

### Human-in-the-Loop

The system is intended to assist screening rather than replace ophthalmologists.

### Confidence Awareness

Calibrated probabilities are provided to improve interpretation of model confidence, but confidence should not be treated as clinical certainty.

### Explainability

Grad-CAM provides model-level visual explanations but does not establish clinical causality.

### Data Protection

Production deployments should implement:

* Secure data transmission
* Authentication and authorization
* Encryption
* Access control
* Audit logging
* Secure temporary storage
* Automatic deletion of temporary files
* Data minimization
* De-identification where appropriate

---

## Current Prototype Scope

The current working prototype focuses on the complete **AI screening and explainability loop**:

```text
                 CURRENT PROTOTYPE
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
 Image Quality      AI Model       Explainability
 Assessment         Inference          │
        │              │               │
        │          EfficientNet        │
        │              │            Grad-CAM
        │             TTA               │
        │              │               │
        └──────────────┼───────────────┘
                       ▼
              Calibrated Results
                       │
                       ▼
                PDF Screening
                   Report
```

This provides a functional foundation that can be extended with more advanced retinal analysis and healthcare workflow simulation.

---

## Model Training

Training and experimentation are separated from the inference runtime.

Training code:

```text
backend/train_improved.py
```

Run from the project root:

```bash
python backend/train_improved.py
```

The training pipeline can be independently modified for:

* Hyperparameter tuning
* Loss-function experimentation
* Dataset changes
* Model fine-tuning
* Evaluation

---

## Evaluation

Model performance should be evaluated using more than accuracy.

Recommended metrics include:

| Metric               | Purpose                                |
| -------------------- | -------------------------------------- |
| Accuracy             | Overall classification performance     |
| Precision            | Reliability of positive predictions    |
| Recall / Sensitivity | Ability to detect relevant cases       |
| Specificity          | Ability to identify non-cases          |
| F1 Score             | Balance between precision and recall   |
| Confusion Matrix     | Class-level error analysis             |
| ROC-AUC              | Discrimination performance             |
| PR-AUC               | Performance under class imbalance      |
| Calibration Error    | Reliability of predicted probabilities |

For a medical screening system, **sensitivity, specificity, calibration, robustness, and external validation** are especially important.

---

## Prototype Limitations

The current prototype should not be considered clinically validated.

Performance may vary due to:

* Dataset size
* Dataset diversity
* Class imbalance
* Image acquisition conditions
* Camera characteristics
* Annotation quality
* Population differences
* Distribution shift

Performance on a particular dataset does not guarantee equivalent performance in hospitals, clinics, or field environments.

External validation and clinical evaluation are required before real-world deployment.

---

## Future Development

The current prototype is intentionally designed as a modular foundation for the broader SIH solution.

Planned extensions include:

* Advanced retinal structure segmentation
* Optic disc localization
* Fovea localization
* Retinal vessel segmentation
* Microaneurysm detection
* Exudate segmentation
* Hemorrhage detection
* Neovascularization detection
* Lesion-level clinical evidence
* Advanced image enhancement
* External clinical validation
* Telemedicine workflow simulation
* Resource allocation optimization

---

# SIH Alignment & Future Integration

The broader SIH problem calls for a deployment-oriented retinal screening system capable of handling real-world challenges such as variable image quality, explainability, clinical validation, and large-scale telemedicine workflows.

The current prototype establishes the core AI screening layer.

The next development stage will extend the system toward the complete SIH vision using **MATLAB and Simulink**.

### MATLAB Integration

MATLAB-based development is planned for advanced retinal image processing and analysis, including:

```text
Image Quality Assessment
        │
        ▼
Adaptive Enhancement
        │
        ▼
Retinal Structure Analysis
        │
        ▼
Lesion Detection
        │
        ▼
Clinical Evidence
```

Potential components include:

* CLAHE-based enhancement
* Illumination normalization
* Advanced denoising
* Field-of-view assessment
* Retinal vessel segmentation
* Optic disc localization
* Fovea localization
* Microaneurysm detection
* Exudate segmentation
* Hemorrhage detection
* Neovascularization analysis

### Simulink Integration

Simulink will be used in the planned system-level simulation of the telemedicine screening workflow.

Potential parameters include:

* Fundus image acquisition rate
* Network bandwidth
* Image transmission time
* AI processing throughput
* Server capacity
* Ophthalmologist review capacity
* Patient queue size
* Referral rate
* District-level screening workload

The objective is to study how the AI-assisted workflow could scale to large screening programs and optimize resource allocation.

### SIH Performance Targets

The original SIH problem specifies target performance for referable DR of:

* **Sensitivity > 90%**
* **Specificity > 85%**

These are treated as **target validation criteria for the extended system**, not as claimed performance of the current prototype.

The extended solution will require appropriate benchmark evaluation, external validation, and clinical assessment before these targets can be considered achieved.

---

## Project Vision

The project is designed to evolve from a functional AI prototype into a broader **quality-aware, explainable, clinically evaluable, and scalable DR screening platform**.

```text
CURRENT
Python AI Screening Prototype
        │
        ▼
MATLAB Retinal Image Analysis
        │
        ▼
Lesion-Level Clinical Evidence
        │
        ▼
Advanced Explainable Screening
        │
        ▼
Simulink Telemedicine Simulation
        │
        ▼
Resource Optimization
        │
        ▼
Scalable District-Level Screening
```

The immediate focus is on building a reliable and demonstrable AI screening prototype, while the architecture provides a clear path toward the complete SIH problem requirements.

---

## License

This project is distributed under the terms specified in:

```text
MIT LICENSE
```

---

## Acknowledgement

Developed as a **Smart India Hackathon prototype** exploring explainable artificial intelligence for diabetic retinopathy screening and scalable healthcare applications.

---

> **Check the image → Screen with AI → Calibrate the confidence → Explain the prediction → Generate the report → Keep the clinician in the loop.**
