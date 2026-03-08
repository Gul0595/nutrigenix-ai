
import pdfplumber
import re
from dataclasses import dataclass
from typing import List

# ─────────────────────────────────────────────
# Biomarker Data Object
# ─────────────────────────────────────────────

@dataclass
class Biomarker:
    name: str
    value: float


@dataclass
class ExtractionResult:
    biomarkers: List[Biomarker]
    confidence_score: float


# ─────────────────────────────────────────────
# Biomarker Extractor
# ─────────────────────────────────────────────

class BiomarkerExtractor:

    # Common biomarker patterns
    BIOMARKER_PATTERNS = {
        "hemoglobin": r"(hemoglobin|hgb)\s*[:\-]?\s*(\d+\.?\d*)",
        "ferritin": r"(ferritin)\s*[:\-]?\s*(\d+\.?\d*)",
        "glucose": r"(glucose)\s*[:\-]?\s*(\d+\.?\d*)",
        "cholesterol": r"(cholesterol)\s*[:\-]?\s*(\d+\.?\d*)",
        "alt": r"(alt|sgpt)\s*[:\-]?\s*(\d+\.?\d*)",
        "uric_acid": r"(uric\s*acid)\s*[:\-]?\s*(\d+\.?\d*)",
        "vitamin_d": r"(vitamin\s*d|25-oh\s*d)\s*[:\-]?\s*(\d+\.?\d*)",
        "vitamin_b12": r"(vitamin\s*b12|b12)\s*[:\-]?\s*(\d+\.?\d*)"
    }

    def extract(self, pdf_path: str) -> ExtractionResult:

        text = ""

        # Extract text from PDF
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"

        text = text.lower()

        biomarkers = []

        # Search biomarkers
        for biomarker, pattern in self.BIOMARKER_PATTERNS.items():

            match = re.search(pattern, text)

            if match:
                value = float(match.group(2))

                biomarkers.append(
                    Biomarker(
                        name=biomarker,
                        value=value
                    )
                )

        # Confidence score based on number extracted
        confidence = len(biomarkers) / len(self.BIOMARKER_PATTERNS)

        return ExtractionResult(
            biomarkers=biomarkers,
            confidence_score=confidence
        )