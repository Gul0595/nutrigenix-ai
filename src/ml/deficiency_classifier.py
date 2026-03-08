
"""
Deficiency Classifier — Training + Prediction Interface
"""

import pickle
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List

import shap
import mlflow
import mlflow.sklearn

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from loguru import logger


DATA_DIR = Path("data/raw")
MODEL_DIR = Path("models/deficiency_classifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Prediction object used by demo.py
# ─────────────────────────────────────────────
@dataclass
class DeficiencyPrediction:
    deficiency: str
    probability: float
    severity: str
    top_features: list
    rank: int

    def to_dict(self):
        return {
            "deficiency": self.deficiency,
            "probability": self.probability,
            "severity": self.severity,
            "top_features": self.top_features,
            "rank": self.rank,
        }


# ─────────────────────────────────────────────
# Classifier used by demo.py
# ─────────────────────────────────────────────
class DeficiencyClassifier:

    def __init__(self):

        model_file = MODEL_DIR / "models.pkl"

        if not model_file.exists():
            raise FileNotFoundError("Models not trained")

        with open(model_file, "rb") as f:
            self.models = pickle.load(f)

    def predict(self, biomarkers: dict) -> List[DeficiencyPrediction]:

        df = pd.DataFrame([biomarkers])

        results = []

        for target, info in self.models.items():

            model = info["model"]
            features = info["features"]

            X = df.reindex(columns=features).fillna(0)

            prob = float(model.predict_proba(X)[0][1])

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

        results.sort(key=lambda x: x.probability, reverse=True)

        for i, r in enumerate(results):
            r.rank = i + 1

        return results


# ─────────────────────────────────────────────
# Training pipeline
# ─────────────────────────────────────────────
def load_data():

    logger.info("Loading NHANES datasets...")

    bio = pd.read_sas(DATA_DIR / "nhanes_biochemistry.xpt", format="xport")
    iron = pd.read_sas(DATA_DIR / "nhanes_iron.xpt", format="xport")
    demo = pd.read_sas(DATA_DIR / "nhanes_demographics.xpt", format="xport")
    cbc = pd.read_sas(DATA_DIR / "nhanes_cbc.xpt", format="xport")

    df = demo.merge(bio, on="SEQN", how="left")
    df = df.merge(iron, on="SEQN", how="left")
    df = df.merge(cbc, on="SEQN", how="left")

    logger.info(f"Merged dataset: {df.shape}")

    return df


def build_features(df):

    features = pd.DataFrame()
    labels = pd.DataFrame()

    features["age"] = df.get("RIDAGEYR")
    features["sex_male"] = (df.get("RIAGENDR") == 1).astype(float)
    features["bmi"] = df.get("BMXBMI")

    features["glucose"] = df.get("LBXSGL")
    features["creatinine"] = df.get("LBXSCR")
    features["cholesterol"] = df.get("LBXSCH")
    features["triglycerides"] = df.get("LBXSTR")

    features["alt"] = df.get("LBXSATSI")
    features["ast"] = df.get("LBXSASSI")

    features["ferritin"] = df.get("LBXFER")
    features["hemoglobin"] = df.get("LBXHGB")
    features["uric_acid"] = df.get("LBXSUA")

    labels["iron_deficient"] = (features["ferritin"] < 12).astype(int)
    labels["anemic"] = (features["hemoglobin"] < 12).astype(int)
    labels["high_glucose"] = (features["glucose"] > 100).astype(int)
    labels["high_cholesterol"] = (features["cholesterol"] > 200).astype(int)
    labels["elevated_liver"] = (features["alt"] > 40).astype(int)
    labels["high_uric_acid"] = (features["uric_acid"] > 7).astype(int)

    features = features.fillna(features.median())

    return features, labels


def train(features, labels):

    mlflow.set_experiment("nutrigenix_deficiency")

    models = {}

    for target in labels.columns:

        y = labels[target]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="logloss"
            ))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = cross_val_score(model, features, y, cv=cv, scoring="accuracy")

        logger.info(f"{target} accuracy: {scores.mean():.3f}")

        model.fit(features, y)

        models[target] = {
            "model": model,
            "features": list(features.columns)
        }

    with open(MODEL_DIR / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    logger.success(f"Saved {len(models)} models")


if __name__ == "__main__":

    logger.info("Training deficiency models")

    df = load_data()

    features, labels = build_features(df)

    train(features, labels)

    logger.info("Training complete")