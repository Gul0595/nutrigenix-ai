from pathlib import Path
import pickle
import pandas as pd
from typing import List
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Prediction Output Schema
# ─────────────────────────────────────────────
@dataclass
class DeficiencyPrediction:
    deficiency: str
    probability: float
    severity: str
    top_features: list
    rank: int


# ─────────────────────────────────────────────
# Locate Project Root
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


# ─────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────
class DeficiencyClassifier:

    def __init__(self):

        model_file = MODEL_DIR / "models.pkl"

        print("Loading ML models...")
        print("Model path:", model_file)

        if not model_file.exists():
            raise FileNotFoundError(f"Models not found at {model_file}")

        with open(model_file, "rb") as f:
            self.models = pickle.load(f)

        print("Models loaded successfully")

    # ─────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────
    def predict(self, biomarkers: dict) -> List[DeficiencyPrediction]:

        df = pd.DataFrame([biomarkers])

        results = []

        for target, info in self.models.items():

            model = info["model"]
            features = info["features"]

            # ensure correct feature order
            X = df.reindex(columns=features).fillna(0)

            prob = float(model.predict_proba(X)[0][1])

            # severity logic
            if prob > 0.75:
                severity = "severe"
            elif prob > 0.5:
                severity = "moderate"
            else:
                severity = "mild"

            results.append(
                DeficiencyPrediction(
                    deficiency=target,
                    probability=prob,
                    severity=severity,
                    top_features=[],
                    rank=0
                )
            )

        # rank by probability
        results.sort(key=lambda x: x.probability, reverse=True)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results
