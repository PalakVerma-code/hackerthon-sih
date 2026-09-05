import base64
import datetime
import io

import requests
import streamlit as st
import streamlit.components.v1 as components
from fpdf import FPDF
from PIL import Image


# -------------------------------------------------------------
# PDF Generator Function
# -------------------------------------------------------------
def generate_pdf_report(
    patient_name,
    patient_age,
    patient_id,
    diagnosis,
    grade,
    action,
    rec,
    orig_img,
    heatmap_img,
    is_referable=None,
    referable_prob=None,
    needs_review=None,
    confidence=None,
):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Diabetic Retinopathy Screening Report", ln=True, align="C")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(
        0,
        5,
        f"Date: {datetime.datetime.now().strftime('%d-%m-%Y %H:%M')}",
        ln=True,
        align="C",
    )
    pdf.ln(10)

    # Patient Information
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Patient Information:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(0, 6, f"Patient ID: {patient_id}", ln=True)
    pdf.cell(0, 6, f"Age/Gender: {patient_age}", ln=True)
    pdf.ln(8)

    # Diagnostic Findings
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "AI Diagnostic Findings:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 6, f"Diagnosis: {diagnosis} (Severity Grade {grade})", ln=True)
    if confidence is not None:
        pdf.cell(
            0, 6, f"Model Confidence (uncalibrated): {confidence * 100:.1f}%", ln=True
        )
    if referable_prob is not None:
        pdf.cell(
            0, 6, f"Referable DR Probability: {referable_prob * 100:.1f}%", ln=True
        )
    pdf.cell(0, 6, f"Action Level: {action}", ln=True)

    if is_referable:
        pdf.set_text_color(200, 0, 0)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(
            0,
            6,
            "REFERABLE - specialist referral recommended regardless of exact grade.",
            ln=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)
    if needs_review:
        pdf.set_text_color(200, 0, 0)
        pdf.set_font("Arial", "B", 10)
        pdf.cell(
            0,
            6,
            "FLAGGED FOR MANDATORY HUMAN REVIEW ",
            ln=True,
        )
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 10)

    pdf.ln(5)

    # Clinical Recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Clinical Recommendation:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 6, rec)
    pdf.ln(10)

    # Visual Scans
    orig_img.save("temp_orig.jpg")
    heatmap_img.save("temp_heat.jpg")

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visual Fundus & XAI Heatmap Analysis:", ln=True)
    pdf.image("temp_orig.jpg", x=15, y=pdf.get_y() + 5, w=80)
    pdf.image("temp_heat.jpg", x=105, y=pdf.get_y() + 5, w=80)

    pdf_bytes = pdf.output(dest="S").encode("latin-1")
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    return pdf_bytes, pdf_base64


# -------------------------------------------------------------
# Streamlit App Layout
# -------------------------------------------------------------
st.set_page_config(page_title="SIH 2026 - DR Screening & Report", layout="wide")
st.title("👁️ Explainable AI for Diabetic Retinopathy Screening")

# Sidebar
st.sidebar.header("📋 Patient Details")
p_name = st.sidebar.text_input("Patient Name", "Ramesh Kumar")
p_id = st.sidebar.text_input("Patient ID", "PHC-2026-089")
p_age = st.sidebar.text_input("Age / Gender", "54 / Male")

uploaded_file = st.file_uploader("Upload Fundus Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Uploaded Original Fundus Scan", width="stretch")

    if st.button("Run AI Diagnosis"):
        with st.spinner("Processing diagnosis and generating report..."):
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type,
                )
            }
            response = requests.post("http://127.0.0.1:8000/predict", files=files)

            if response.status_code == 422:
                data = response.json()
                with col2:
                    st.error(
                        f"❌ Image Rejected: {data.get('reason', 'quality insufficient')}"
                    )
                    with st.expander("Quality metrics"):
                        st.json(data.get("quality_metrics", {}))

            elif response.status_code == 200:
                data = response.json()

                img_bytes = base64.b64decode(data["heatmap_base64"])
                heatmap_img = Image.open(io.BytesIO(img_bytes))

                is_referable = data.get("is_referable")
                referable_prob = data.get("referable_probability")
                needs_review = data.get("needs_human_review")
                confidence = data.get("confidence")
                uncertainty = data.get("uncertainty")
                class_probs = data.get("class_probabilities")
                enhancement_applied = data.get("enhancement_applied")

                with col2:
                    if enhancement_applied:
                        st.info(
                            "ℹ️ Image quality was borderline -- automatic enhancement applied before grading."
                        )

                    if needs_review:
                        st.warning(
                            "⚠️ Low confidence assessment with significant reviewer disagreement - mandatory ophthalmologist review required."
                        )

                    if is_referable is not None:
                        if is_referable:
                            st.error(
                                f"🔴 REFERABLE (P={referable_prob * 100:.1f}%) -- recommend referral."
                            )
                        else:
                            st.success(
                                f"🟢 Not flagged referable (P={referable_prob * 100:.1f}%)."
                            )

                    st.success(
                        f"**Diagnosis:** {data['diagnosis']} (Grade {data['severity_grade']})"
                    )
                    if confidence is not None:
                        st.write(f"Model confidence : {confidence * 100:.1f}%")
                    st.warning(f"**Action Required:** {data['action_required']}")
                    st.info(
                        f"**Clinical Recommendation:** {data['clinical_recommendation']}"
                    )
                    st.image(
                        heatmap_img,
                        caption="Grad-CAM Heatmap Analysis",
                        width="stretch",
                    )

                    if class_probs:
                        with st.expander(
                            "Model uncertainty & full probability distribution"
                        ):
                            st.write(f"TTA disagreement (uncertainty): {uncertainty}")
                            grade_labels = [
                                "No DR",
                                "Mild",
                                "Moderate",
                                "Severe",
                                "Proliferative",
                            ]
                            st.bar_chart(dict(zip(grade_labels, class_probs)))

                    # Generate PDF
                    pdf_bytes, pdf_b64 = generate_pdf_report(
                        p_name,
                        p_age,
                        p_id,
                        data["diagnosis"],
                        data["severity_grade"],
                        data["action_required"],
                        data["clinical_recommendation"],
                        image,
                        heatmap_img,
                        is_referable=is_referable,
                        referable_prob=referable_prob,
                        needs_review=needs_review,
                        confidence=confidence,
                    )

                    st.download_button(
                        label="📄 Download Medical Report (PDF)",
                        data=pdf_bytes,
                        file_name=f"DR_Report_{p_id}.pdf",
                        mime="application/pdf",
                    )

                    # ---------------------------------------------------------
                    # UPDATED PREVIEW (Fixed Base64 HTML Embed Tag)
                    # ---------------------------------------------------------
                    st.markdown("---")
                    st.subheader("👁️ Live Medical Report Preview")

                    pdf_display = f"""
                        <embed src="data:application/pdf;base64,{pdf_b64}"
                               width="100%" height="650" type="application/pdf">
                    """
                    st.markdown(pdf_display, unsafe_allow_html=True)

            else:
                st.error("Backend Connection Failed!")
