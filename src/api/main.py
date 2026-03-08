from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from src.ml.deficiency_classifier import DeficiencyClassifier
from src.agents.supplement_agent import SupplementAgent


# -----------------------------------------------------
# FastAPI App
# -----------------------------------------------------

app = FastAPI(
    title="NutriGenix AI",
    description="AI-powered biomarker deficiency detection",
    version="1.0"
)


# -----------------------------------------------------
# Load AI models ONCE at startup
# -----------------------------------------------------

print("Loading ML models...")
classifier = DeficiencyClassifier()
supplement_agent = SupplementAgent()
print("Models loaded successfully")


# -----------------------------------------------------
# Input Schema
# -----------------------------------------------------

class Biomarkers(BaseModel):
    age: float
    bmi: float
    ferritin: float
    hemoglobin: float
    glucose: float
    cholesterol: float
    alt: float
    uric_acid: float


# -----------------------------------------------------
# Root Endpoint
# -----------------------------------------------------

@app.get("/")
def root():
    return {"message": "NutriGenix API running"}


# -----------------------------------------------------
# Prediction Endpoint
# -----------------------------------------------------

@app.post("/predict")
def predict(biomarkers: Biomarkers):

    try:

        print("Received API request")

        data = biomarkers.dict()

        print("Running deficiency classifier")
        predictions = classifier.predict(data)

        deficiencies = [
            p.deficiency for p in predictions
            if p.probability > 0.5
        ]

        print("Generating supplement protocol")
        supplements = supplement_agent.generate_protocol(deficiencies)

        print("Returning response")

        return {
            "deficiencies": [p.to_dict() for p in predictions],
            "supplements": [s.__dict__ for s in supplements]
        }

    except Exception as e:

        print("Prediction error:", str(e))

        return {
            "error": "Prediction failed",
            "details": str(e)
        }
