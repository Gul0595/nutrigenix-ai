
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.biomarker_extractor import BiomarkerExtractor

extractor = BiomarkerExtractor()

# path to sample PDF
pdf_path = "data/sample_report.pdf"

result = extractor.extract(pdf_path)

for biomarker in result.biomarkers:
    print(biomarker.name, biomarker.value)

print("Confidence:", result.confidence_score)
