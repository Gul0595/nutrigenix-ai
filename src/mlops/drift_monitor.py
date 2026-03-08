"""
MLOps: Data Drift Monitor
==========================
Uses Evidently AI to detect when incoming biomarker distributions
drift from the training distribution (NHANES baseline).
Logs drift scores to MLflow and triggers alerts.
"""

import json
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import DatasetDriftMetric
from loguru import logger


class DriftMonitor:
    """
    Monitors incoming biomarker data for distribution drift.
    Compares live data against NHANES training distribution.
    """

    DRIFT_ALERT_THRESHOLD = 0.15  # Alert if drift score > 15%
    BUFFER_SIZE = 100             # Accumulate 100 requests before running drift check

    def __init__(self, model_dir: str = "./models/deficiency_classifier"):
        self.model_dir = Path(model_dir)
        self.buffer: list[dict] = []
        self.reference_stats_path = self.model_dir / "reference_stats.pkl"
        self.reference_df: pd.DataFrame | None = None
        self._load_reference()

    def _load_reference(self):
        """Load NHANES training distribution stats."""
        if self.reference_stats_path.exists():
            with open(self.reference_stats_path, "rb") as f:
                self.reference_df = pickle.load(f)
            logger.info("Drift monitor: Loaded reference distribution")
        else:
            logger.warning("Drift monitor: No reference distribution found. Run save_reference() after training.")

    def save_reference(self, df: pd.DataFrame):
        """Call this after training to save the reference distribution."""
        sample = df.sample(min(1000, len(df)), random_state=42)
        with open(self.reference_stats_path, "wb") as f:
            pickle.dump(sample, f)
        self.reference_df = sample
        logger.info(f"Drift monitor: Saved reference distribution ({len(sample)} samples)")

    def record(self, biomarkers: dict, request_id: str):
        """
        Record incoming biomarker data. Runs drift check every BUFFER_SIZE requests.
        Called as a background task from the API.
        """
        self.buffer.append({**biomarkers, "request_id": request_id, "timestamp": datetime.utcnow().isoformat()})

        if len(self.buffer) >= self.BUFFER_SIZE:
            self._run_drift_check()
            self.buffer = []

    def _run_drift_check(self):
        """Run Evidently drift analysis and log to MLflow."""
        if self.reference_df is None:
            logger.warning("Drift check skipped: no reference distribution")
            return

        current_df = pd.DataFrame(self.buffer)

        # Keep only numeric biomarker columns
        numeric_cols = [c for c in current_df.columns
                        if c not in ("request_id", "timestamp")
                        and current_df[c].dtype in (float, int)]

        if not numeric_cols:
            return

        # Align columns with reference
        common_cols = [c for c in numeric_cols if c in self.reference_df.columns]
        if len(common_cols) < 3:
            logger.warning(f"Drift check: only {len(common_cols)} common columns, skipping")
            return

        ref = self.reference_df[common_cols].dropna()
        cur = current_df[common_cols].fillna(current_df[common_cols].median())

        try:
            # Run Evidently drift report
            report = Report(metrics=[
                DataDriftPreset(),
                DatasetDriftMetric(),
            ])
            report.run(reference_data=ref, current_data=cur)

            results = report.as_dict()
            drift_score = results["metrics"][1]["result"]["drift_score"]
            n_drifted = results["metrics"][1]["result"]["number_of_drifted_columns"]
            share_drifted = results["metrics"][1]["result"]["share_of_drifted_columns"]

            logger.info(
                f"Drift check: score={drift_score:.3f}, "
                f"drifted_columns={n_drifted}/{len(common_cols)} ({share_drifted:.1%})"
            )

            # Log to MLflow
            with mlflow.start_run(run_name=f"drift_check_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_metric("drift_score", drift_score)
                mlflow.log_metric("drifted_columns", n_drifted)
                mlflow.log_metric("share_drifted", share_drifted)
                mlflow.log_metric("sample_size", len(cur))

                # Save HTML report as artifact
                report_path = f"/tmp/drift_report_{datetime.utcnow().strftime('%Y%m%d')}.html"
                report.save_html(report_path)
                mlflow.log_artifact(report_path)

            # Alert if drift exceeds threshold
            if drift_score > self.DRIFT_ALERT_THRESHOLD:
                self._trigger_alert(drift_score, n_drifted, common_cols)

        except Exception as e:
            logger.error(f"Drift check failed: {e}")

    def _trigger_alert(self, drift_score: float, n_drifted: int, cols: list):
        """Log alert when drift is detected. In production: send to Slack/email."""
        alert = {
            "type": "data_drift_alert",
            "timestamp": datetime.utcnow().isoformat(),
            "drift_score": round(drift_score, 4),
            "drifted_columns": n_drifted,
            "threshold": self.DRIFT_ALERT_THRESHOLD,
            "action_required": "Review incoming data distribution. Consider model retraining.",
        }
        logger.warning(f"🚨 DATA DRIFT DETECTED: {json.dumps(alert)}")

        # Write to alert log (in production, integrate with PagerDuty/Slack)
        alert_path = Path("./data/alerts.jsonl")
        with open(alert_path, "a") as f:
            f.write(json.dumps(alert) + "\n")
