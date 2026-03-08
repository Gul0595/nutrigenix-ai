"""
scripts/demo.py
================
End-to-end demo of the NutriGenix pipeline.
Run with: python scripts/demo.py
Or with a real PDF: python scripts/demo.py --pdf path/to/blood_report.pdf
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from loguru import logger

# ── Mock biomarker data (simulates a real blood report extraction) ─────────────
DEMO_BIOMARKERS = {
    # ↓ Deficient values to trigger interesting recommendations
    "vitamin_d": 14.5,       # Very low (normal: 30-100 ng/mL)
    "vitamin_b12": 185.0,    # Low (normal: 200-900 pg/mL)
    "ferritin": 8.0,         # Low (normal: 12-300 ng/mL)
    "magnesium": 1.5,        # Low (normal: 1.7-2.2 mg/dL)
    "folate": 2.8,           # Low (normal: 3.4-20 ng/mL)
    "hemoglobin": 11.2,      # Low (normal: 12-17.5 g/dL)

    # Normal values
    "glucose": 92.0,
    "hba1c": 5.4,
    "cholesterol": 185.0,
    "triglycerides": 120.0,
    "hdl": 52.0,
    "ldl": 95.0,
    "creatinine": 0.9,
    "alt": 28.0,
    "ast": 24.0,
    "tsh": 3.2,
    "crp": 1.8,
    "calcium": 9.2,
    "zinc": 55.0,            # Slightly low

    # Patient demographics
    "age": 42,
    "bmi": 26.5,
    "sex_male": 0,           # Female patient
}

DEMO_PATIENT_CONTEXT = {
    "age": 42,
    "sex": "female",
    "weight_kg": 68,
    "conditions": ["hypothyroidism"],          # Has thyroid condition
    "medications": ["levothyroxine"],          # On thyroid medication
}


def run_demo(pdf_path: str = None):
    logger.info("=" * 60)
    logger.info("🧬 NutriGenix Demo — Personalized Supplement Formulation")
    logger.info("=" * 60)

    # ── Layer 1: PDF extraction (or use mock data) ─────────────────────────────
    if pdf_path and Path(pdf_path).exists():
        logger.info(f"\n📄 Layer 1: Extracting biomarkers from {pdf_path}...")
        from src.ingestion.biomarker_extractor import BiomarkerExtractor
        extractor = BiomarkerExtractor()
        extraction = extractor.extract(pdf_path)
        biomarkers = {b.name.lower().replace(" ", "_"): b.value for b in extraction.biomarkers}
        logger.info(f"  Extracted {len(extraction.biomarkers)} biomarkers")
        logger.info(f"  Confidence: {extraction.confidence_score:.0%}")
    else:
        logger.info("\n📊 Layer 1: Using demo biomarker data (no PDF provided)...")
        biomarkers = DEMO_BIOMARKERS
        logger.info(f"  {len(biomarkers)} biomarkers loaded")

    # Show deficient biomarkers
    deficient_display = {k: v for k, v in biomarkers.items()
                         if k not in ("age", "bmi", "sex_male")}
    logger.info(f"\n  Key values:")
    for k, v in list(deficient_display.items())[:8]:
        logger.info(f"    {k:25s}: {v}")

    # ── Layer 2: Deficiency classification ────────────────────────────────────
    logger.info("\n🤖 Layer 2: Running deficiency classifier (XGBoost + SHAP)...")
    from src.ml.deficiency_classifier import DeficiencyClassifier
    clf = DeficiencyClassifier()

    try:
        predictions = clf.predict(biomarkers)
        significant = [p for p in predictions if p.probability > 0.30]
        logger.info(f"  Deficiencies detected ({len(significant)}):")
        for p in significant:
            bar = "█" * int(p.probability * 20)
            logger.info(f"    {p.deficiency:30s} {bar} {p.probability:.0%} ({p.severity})")
            if p.top_features:
                logger.info(f"      → Key driver: {p.top_features[0]['feature']} = {p.top_features[0]['value']}")
    except FileNotFoundError:
        logger.warning("  ⚠️  Models not trained yet. Using mock deficiency predictions.")
        from src.ml.deficiency_classifier import DeficiencyPrediction
        predictions = [
            DeficiencyPrediction("vitamin_d_deficient", 0.94, "severe",
                                 [{"feature": "vitamin_d", "value": 14.5, "direction": "increases_risk", "impact": 0.8}], 1),
            DeficiencyPrediction("iron_deficient", 0.87, "moderate",
                                 [{"feature": "ferritin", "value": 8.0, "direction": "increases_risk", "impact": 0.7}], 1),
            DeficiencyPrediction("vitamin_b12_deficient", 0.78, "moderate",
                                 [{"feature": "vitamin_b12", "value": 185.0, "direction": "increases_risk", "impact": 0.6}], 2),
            DeficiencyPrediction("anemic", 0.71, "moderate",
                                 [{"feature": "hemoglobin", "value": 11.2, "direction": "increases_risk", "impact": 0.7}], 2),
            DeficiencyPrediction("magnesium_deficient", 0.62, "mild",
                                 [{"feature": "magnesium", "value": 1.5, "direction": "increases_risk", "impact": 0.5}], 3),
        ]
        significant = predictions

    deficiencies_dict = [p.to_dict() for p in significant]

    # ── Layer 3: Agentic formulation ──────────────────────────────────────────
    logger.info("\n🔗 Layer 3: Running 5-agent LangGraph formulation pipeline...")
    logger.info("  [1/5] ResearchAgent: Querying PubMed vector store...")
    logger.info("  [2/5] DosingAgent:   Calculating personalized doses...")
    logger.info("  [3/5] SafetyAgent:   Checking drug-supplement interactions...")
    logger.info("  [4/5] FormulationAgent: Assembling final stack...")
    logger.info("  [5/5] AuditAgent:    Generating citations & audit trail...")

    try:
        from src.agents.formulation_pipeline import run_formulation
        result = run_formulation(
            patient_id="DEMO_001",
            biomarkers=biomarkers,
            deficiencies=deficiencies_dict,
            patient_context=DEMO_PATIENT_CONTEXT,
        )

        # ── Display results ────────────────────────────────────────────────────
        logger.info(f"\n✅ FINAL FORMULATION ({len(result['final_formulation'])} supplements):")
        logger.info("─" * 60)
        for i, supp in enumerate(result["final_formulation"], 1):
            logger.info(f"\n  {i}. {supp.get('supplement', 'Unknown')}")
            logger.info(f"     Dose:     {supp.get('recommended_dose', 'N/A')}")
            logger.info(f"     Form:     {supp.get('form', 'N/A')}")
            logger.info(f"     Timing:   {supp.get('timing', 'N/A')}")
            logger.info(f"     Duration: {supp.get('duration', 'N/A')}")
            if supp.get("safety_notes"):
                logger.warning(f"     ⚠️  {supp['safety_notes'][0]}")

        if result.get("warnings"):
            logger.info(f"\n⚠️  WARNINGS:")
            for w in result["warnings"]:
                logger.warning(f"   • {w}")

        if result.get("citations"):
            logger.info(f"\n📚 EVIDENCE CITATIONS ({len(result['citations'])}):")
            for c in result["citations"][:3]:
                logger.info(f"   {c.get('supplement')}: {c.get('evidence_level')} evidence | {len(c.get('pmids', []))} papers")

        logger.info(f"\n📊 CONFIDENCE SCORES:")
        for supp, score in result.get("confidence_scores", {}).items():
            bar = "▓" * int(score * 20)
            logger.info(f"   {supp:30s} {bar} {score:.0%}")

    except Exception as e:
        logger.warning(f"  Agent pipeline requires Ollama running. Error: {e}")
        logger.info("\n  📝 To run the full pipeline:")
        logger.info("     1. Install Ollama: https://ollama.ai")
        logger.info("     2. Pull model: ollama pull mistral:7b-instruct-q4_K_M")
        logger.info("     3. Start: ollama serve")
        logger.info("     4. Re-run this demo")

    logger.info("\n" + "=" * 60)
    logger.info("🎉 NutriGenix Demo Complete!")
    logger.info("   Next steps:")
    logger.info("   • Build vector store:    python scripts/build_vector_store.py")
    logger.info("   • Train ML model:        python src/ml/deficiency_classifier.py")
    logger.info("   • Build knowledge graph: python scripts/build_knowledge_graph.py")
    logger.info("   • Launch API:            uvicorn src.api.main:app --reload")
    logger.info("   • View MLflow:           http://localhost:5000")
    logger.info("   • View Neo4j:            http://localhost:7474")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NutriGenix end-to-end demo")
    parser.add_argument("--pdf", type=str, help="Path to blood report PDF (optional)")
    args = parser.parse_args()
    run_demo(pdf_path=args.pdf)
