import streamlit as st
import requests
import sys
from pathlib import Path

# allow imports from src
sys.path.append(str(Path(__file__).resolve().parent))

from src.ingestion.biomarker_extractor import BiomarkerExtractor

API_URL = "http://127.0.0.1:8001/predict"

st.set_page_config(
    page_title="NutriGenix AI",
    page_icon="🧬",
    layout="centered"
)

st.title("🧬 NutriGenix AI")
st.markdown("AI-powered biomarker deficiency detection")

st.divider()

# ---------------------------------------------------
# Cached Biomarker Extraction
# ---------------------------------------------------

@st.cache_data
def extract_biomarkers(file_path):
    extractor = BiomarkerExtractor()
    result = extractor.extract(file_path)
    return {b.name: b.value for b in result.biomarkers}

# ---------------------------------------------------
# Upload Blood Report
# ---------------------------------------------------

st.subheader("Upload Blood Report")

uploaded_file = st.file_uploader(
    "Upload blood report PDF",
    type=["pdf"]
)

biomarkers = {}

if uploaded_file:

    with open("temp_report.pdf", "wb") as f:
        f.write(uploaded_file.read())

    biomarkers = extract_biomarkers("temp_report.pdf")

    st.success("Biomarkers extracted successfully")

    st.subheader("Extracted Biomarkers")
    st.json(biomarkers)

st.divider()

# ---------------------------------------------------
# Manual Input
# ---------------------------------------------------

st.subheader("Or Enter Biomarkers Manually")

age = st.slider("Age", 18, 80, 40)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

ferritin = st.number_input("Ferritin", min_value=0.0, value=50.0)
hemoglobin = st.number_input("Hemoglobin", min_value=0.0, value=14.0)

glucose = st.number_input("Glucose", min_value=40.0, value=90.0)
cholesterol = st.number_input("Cholesterol", min_value=50.0, value=180.0)

alt = st.number_input("ALT (Liver Enzyme)", min_value=0.0, value=25.0)
uric_acid = st.number_input("Uric Acid", min_value=0.0, value=4.5)

st.divider()

# ---------------------------------------------------
# Run AI Analysis
# ---------------------------------------------------

if st.button("🔍 Analyze Biomarkers"):

    payload = {
        "age": age,
        "bmi": bmi,
        "ferritin": biomarkers.get("ferritin", ferritin),
        "hemoglobin": biomarkers.get("hemoglobin", hemoglobin),
        "glucose": biomarkers.get("glucose", glucose),
        "cholesterol": biomarkers.get("cholesterol", cholesterol),
        "alt": biomarkers.get("alt", alt),
        "uric_acid": biomarkers.get("uric_acid", uric_acid)
    }

    try:

        with st.spinner("Analyzing biomarkers..."):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=5
            )

        if response.status_code == 200:

            data = response.json()

            # --------------------------------
            # Deficiency Predictions
            # --------------------------------

            st.subheader("🧠 Deficiency Predictions")

            for r in data["deficiencies"]:

                name = r["deficiency"].replace("_", " ").title()
                prob = r["probability"]
                severity = r["severity"]

                st.markdown(f"### {name}")
                st.progress(prob)

                st.metric(
                    label="Risk Probability",
                    value=f"{prob:.2%}"
                )

                if severity == "severe":
                    st.error("⚠️ Severe risk detected")
                elif severity == "moderate":
                    st.warning("Moderate risk detected")
                else:
                    st.success("Low risk")

                st.divider()

            # --------------------------------
            # Supplement Protocol
            # --------------------------------

            if data.get("supplements"):

                st.subheader("💊 Recommended Supplements")

                for s in data["supplements"]:

                    st.markdown(f"### {s['name']}")

                    st.write(f"**Dose:** {s['dose']}")
                    st.write(f"**Reason:** {s['reason']}")

                    if s.get("evidence"):

                        st.markdown("**Scientific Evidence:**")

                        for paper in s["evidence"]:
                            st.write(
                                f"PMID {paper['pmid']} — {paper['title']}"
                            )

                    st.divider()

        else:
            st.error("API returned an error")
            st.text(response.text)

    except Exception as e:
        st.error("Could not connect to prediction API")
        st.text(str(e))


st.caption("NutriGenix AI • Personalized Biomarker Intelligence")