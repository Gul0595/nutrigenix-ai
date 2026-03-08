# 🧬 NutriGenix — Personalized Supplement Formulation AI

> End-to-end AI platform that ingests patient biomarker reports (blood work PDFs),
> reasons over nutritional science literature, and generates evidence-based personalized
> supplement formulations with a full clinical audit trail.

## Architecture Overview

```
Blood Report PDF
      │
      ▼
┌─────────────────┐
│  LAYER 1        │  LayoutLMv3 + PyMuPDF + Tesseract OCR
│  PDF Extraction │  → Structured biomarker JSON + LOINC codes
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAYER 2        │  XGBoost deficiency classifier (NHANES trained)
│  ML Reasoning   │  → Deficiency probabilities + risk scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAYER 3        │  LangGraph 5-agent pipeline
│  Formulation    │  Mistral-7B (4-bit quant) + ChromaDB RAG
│  Agents         │  → Evidence-based supplement stack
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAYER 4        │  Audit trail + citation tracer
│  PDF Report     │  → Clinical-grade report (WeasyPrint)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LAYER 5        │  MLflow + BentoML + Evidently AI
│  MLOps          │  → Monitoring, versioning, serving
└─────────────────┘
```

## Tech Stack (100% Free & Open Source)

| Component | Tool | Purpose |
|-----------|------|---------|
| PDF Extraction | LayoutLMv3, PyMuPDF, Tesseract | Biomarker extraction from any lab format |
| ML Model | XGBoost + SHAP | Deficiency classification on NHANES data |
| LLM | Mistral-7B-Instruct (4-bit, Ollama) | Local reasoning, zero API cost |
| Fine-tuning | LoRA/PEFT (HuggingFace) | Nutrition-specific LLM adaptation |
| Vector Store | ChromaDB | 500k PubMed paper embeddings |
| Agents | LangGraph | 5-agent formulation pipeline |
| Knowledge Graph | Neo4j Community | Supplement-drug interaction graph |
| MLOps | MLflow + BentoML + Evidently AI | Tracking, serving, monitoring |
| Data Versioning | DVC | Reproducible data pipelines |
| Data Quality | Great Expectations | Biomarker input validation |
| API | FastAPI | B2B SaaS endpoint |
| Reports | WeasyPrint | Clinical PDF generation |
| Containers | Docker + Docker Compose | Full stack orchestration |

## Quick Start

```bash
# 1. Clone and setup environment
git clone https://github.com/yourusername/nutrigenix.git
cd nutrigenix
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Pull Mistral-7B locally (one-time, ~4GB)
ollama pull mistral:7b-instruct-q4_K_M

# 3. Start all services (Neo4j, ChromaDB, MLflow)
docker-compose up -d

# 4. Download and prepare NHANES data
python scripts/download_nhanes.py

# 5. Build PubMed vector store (first run ~30 min)
python scripts/build_vector_store.py

# 6. Train deficiency classifier
python src/ml/train_deficiency_model.py

# 7. Launch API
uvicorn src.api.main:app --reload

# 8. Try a demo
python scripts/demo.py --pdf data/samples/sample_blood_report.pdf
```

## Project Structure

```
nutrigenix/
├── src/
│   ├── ingestion/          # Layer 1: PDF extraction pipeline
│   ├── ml/                 # Layer 2: Deficiency ML model
│   ├── agents/             # Layer 3: LangGraph agent system
│   ├── api/                # FastAPI endpoints
│   ├── mlops/              # MLflow, Evidently, BentoML
│   ├── reports/            # PDF report generator
│   └── utils/              # Shared utilities
├── models/                 # Saved model artifacts
├── vector_store/           # ChromaDB persistent storage
├── data/
│   ├── raw/                # NHANES, PubMed raw data
│   ├── processed/          # Cleaned, feature-engineered data
│   └── samples/            # Sample blood reports for testing
├── tests/
├── docker/
├── notebooks/              # EDA and experimentation
├── configs/                # All configuration files
└── scripts/                # Setup and utility scripts
```

## Dataset Sources
- **NHANES 2017-2020**: 50k patients with biomarkers + supplement use (CDC, free)
- **PubMed Central OA**: 500k nutrition papers for RAG (NLM, free)
- **USDA FoodData Central**: Micronutrient database (USDA API, free)
- **NIH ODS**: Supplement dosing reference data (NIH, free)
- **DrugBank Open**: Supplement-drug interactions (free tier)

## GPU Requirements
- Minimum: 6GB VRAM (RTX 4050) — uses 4-bit quantized Mistral-7B via Ollama
- Recommended: 8GB+ VRAM for faster inference
