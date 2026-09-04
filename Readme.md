 # Explainable AI for Diabetic Retinopathy Screening

An end-to-end prototype for screening diabetic retinopathy (DR) from retinal fundus images. The application combines a Streamlit user interface with a FastAPI inference service. It classifies an uploaded image into one of five DR severity levels, produces a Grad-CAM visual explanation, and generates a downloadable PDF screening report.

> **Important medical notice:** This project is intended for research, education, and demonstration purposes only. It is not a medical device and must not be used as a substitute for examination or diagnosis by a qualified ophthalmologist. The model output and recommendations require clinical validation before any real-world use.

## Key Capabilities

- Upload PNG, JPG, or JPEG fundus images through a browser-based interface.
- Capture basic patient information for report generation.
- Classify the image using a five-class EfficientNet-B0 model.
- Display the predicted diagnosis, severity grade, suggested action, and clinical recommendation.
- Generate a Grad-CAM heatmap to show image regions that influenced the prediction.
- Export a PDF report containing patient details, findings, recommendation, original image, and heatmap.
- Preview the generated report in the Streamlit application.
- Use CPU automatically, with CUDA selected when available.

## Project Scope

### In scope

This prototype covers the screening workflow from image upload to explainable result presentation:

1. A user enters patient metadata in the Streamlit sidebar.
2. The user uploads a retinal fundus image.
3. Streamlit sends the image to the FastAPI `/predict` endpoint.
4. The backend preprocesses the image, runs inference, and selects the highest-scoring class.
5. Grad-CAM creates a visual explanation for the predicted class.
6. The API returns structured diagnostic information and a Base64-encoded heatmap.
7. The frontend displays the result and creates a PDF report.

### Current limitations

- There is no authentication, user management, database, audit trail, or persistent case history.
- The API currently exposes only one prediction endpoint.
- Model confidence scores, calibration, uncertainty, and quality checks are not returned.
- The model checkpoint and training dataset provenance are not included in this repository.
- Clinical recommendations are static mappings based on the predicted class, not patient-specific medical advice.
- The frontend assumes that the backend is running locally at `http://127.0.0.1:8000`.
- Uploaded images are processed in memory by the API, but temporary JPEG files are written while generating the PDF.
- The current dependency manifest should include `fpdf2` (or the compatible `fpdf` package) because the frontend imports `from fpdf import FPDF`.

## Technology Stack

| Area | Technology | Purpose |
| --- | --- | --- |
| User interface | Streamlit | Image upload, patient fields, result display, and report preview |
| API | FastAPI | HTTP inference service and JSON response contract |
| API server | Uvicorn | Runs the FastAPI application locally |
| Deep learning | PyTorch | Model execution and device management |
| Model architecture | `timm` EfficientNet-B0 | Five-class retinal image classification |
| Image processing | Pillow, OpenCV, NumPy | Image decoding, resizing, visualization, and conversion |
| Preprocessing | Albumentations | Resize, ImageNet normalization, and tensor conversion |
| Explainability | `pytorch-grad-cam` | Grad-CAM heatmap generation |
| HTTP client | Requests | Frontend-to-backend communication |
| Reporting | FPDF | PDF report generation |
| Model artifact | `best_dr_model.pth` | Trained model weights loaded by the backend |

## Repository Structure

```text
hackerthon/
├── backend/
│   ├── best_dr_model.pth       # Trained EfficientNet-B0 weights
│   └── main.py                 # FastAPI app, preprocessing, inference, Grad-CAM
├── frontend/
│   └── app.py                  # Streamlit UI and PDF report generation
├── requirements.txt            # Python dependencies
├── LICENSE
└── Readme.md
```

## DR Classification Levels

The backend maps the model's numeric output to the following screening categories:

| Grade | Label | Current action mapping |
| ---: | --- | --- |
| 0 | No DR | Routine eye checkup |
| 1 | Mild DR | Follow-up in 6-12 months |
| 2 | Moderate DR | Referral to an ophthalmologist |
| 3 | Severe DR | Urgent specialist referral |
| 4 | Proliferative DR | Immediate emergency referral |

These mappings are application text and should not be interpreted as validated clinical protocols.

## API Reference

### `POST /predict`

Accepts one multipart form-data file:

```text
file: <fundus image>
```

Example response shape:

```json
{
  "diagnosis": "Moderate DR",
  "severity_grade": 2,
  "action_required": "Referral to Ophthalmologist",
  "clinical_recommendation": "...",
  "heatmap_base64": "<base64-encoded JPEG>"
}
```

The image is converted to RGB, resized to `224 x 224`, normalized with ImageNet statistics, and passed to the model. The returned heatmap is a JPEG encoded as Base64.

## Local Setup

### Prerequisites

- Python 3.9 or later
- A virtual environment is recommended.
- The model file `backend/best_dr_model.pth` must be present.
- A CUDA-enabled PyTorch installation is optional; the application falls back to CPU.

### Installation

From the `hackerthon` directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the declared dependencies:

```bash
pip install -r requirements.txt
```

Install the PDF dependency required by `frontend/app.py` if it is not already available:

```bash
pip install fpdf2
```

For production or GPU use, install the PyTorch build appropriate for the target CUDA version by following the official PyTorch installation instructions.

## Running the Application

Start the backend first from the `backend` directory so the relative checkpoint path resolves correctly:

```bash
cd backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In a second terminal, activate the same virtual environment and start the frontend from the project root:

```bash
cd ..
streamlit run frontend/app.py
```

Open the URL printed by Streamlit, upload a fundus image, enter patient details, and select **Run AI Diagnosis**.

The FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## Processing Flow

```text
Fundus image
	|
	v
Streamlit frontend -- multipart upload --> FastAPI backend
									|
									v
						RGB conversion and preprocessing
									|
									v
						    EfficientNet-B0 inference
									|
									+--> DR grade and recommendation
									|
									+--> Grad-CAM heatmap
									|
									v
					JSON response with Base64 heatmap
	|
	v
Result display, report preview, and PDF download
```

## Quality, Safety, and Deployment Considerations

Before any clinical or public deployment, the system should have:

- Evaluation on a representative, independently held-out dataset.
- Sensitivity, specificity, AUROC, F1, confusion matrix, and subgroup analysis.
- Image-quality detection and clear handling of ungradable images.
- Model versioning, reproducible training metadata, and dataset documentation.
- Secure transport, authentication, authorization, input validation, and rate limiting.
- Removal or protection of personally identifiable patient information.
- Structured logging, monitoring, audit records, and failure alerts.
- Clinical review, prospective validation, regulatory assessment, and human oversight.

## Future Improvements

### Model and clinical workflow

- Return class probabilities, confidence thresholds, and an explicit “requires review” state.
- Add fundus-image quality assessment before classification.
- Evaluate and calibrate performance across cameras, lighting conditions, regions, age groups, and other relevant subgroups.
- Support multi-image examinations for both eyes and longitudinal comparison.
- Add detection or segmentation of lesions such as microaneurysms, hemorrhages, and exudates.
- Track model versions and include the model version in every report.

### Backend and platform

- Add request validation, file-size limits, content-type checks, and consistent error responses.
- Move model loading and Grad-CAM initialization into application startup so they are not recreated per request.
- Add automated tests for preprocessing, API responses, class mappings, and invalid uploads.
- Add authentication, role-based access, encrypted storage, audit logging, and configurable backend URLs.
- Containerize the services and add CI/CD with security and quality checks.
- Add health and readiness endpoints for deployment monitoring.

### Frontend and reporting

- Add clear loading, timeout, backend-unavailable, and invalid-image states.
- Replace default patient values with a validated form and explicit consent workflow.
- Improve PDF generation with secure temporary-file handling and a unique report identifier.
- Add report history and export formats such as JSON and CSV.
- Provide accessible, multilingual, and responsive clinical workflows.

## Contributing

Contributions should remain focused, document behavioral changes, and include tests where practical. Changes affecting model predictions, clinical wording, patient data, or report output should include an explanation of validation and known limitations.

## License

See [LICENSE](LICENSE) for the terms governing this project.
