"""
Layer 3: LangGraph 5-Agent Formulation Pipeline
================================================
Agents:
  1. ResearchAgent    — RAG over PubMed papers (ChromaDB)
  2. DosingAgent      — Evidence-based dose calculation
  3. SafetyAgent      — Drug-supplement interaction check (Neo4j)
  4. FormulationAgent — Assembles final supplement stack
  5. AuditAgent       — Generates citations + confidence scores

All LLM calls use local Mistral-7B via Ollama (zero API cost, fits 6GB VRAM).
"""

import json
import operator
from typing import Annotated, TypedDict, Optional

from langchain_ollama import OllamaLLM as Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from neo4j import GraphDatabase
from loguru import logger


# ── Agent State (passed between all agents) ──────────────────────────────────
class FormulationState(TypedDict):
    # Inputs
    patient_id: str
    biomarkers: dict                    # Raw biomarker values
    deficiencies: list[dict]            # From ML layer (deficiency predictions)
    patient_context: dict               # Age, sex, conditions, current meds

    # Intermediate state
    research_findings: Annotated[list, operator.add]   # PubMed evidence per deficiency
    safety_flags: Annotated[list, operator.add]        # Drug-supplement conflicts
    draft_formulation: list[dict]                      # Initial supplement recommendations
    dosing_adjustments: dict                           # Dose modifications

    # Output
    final_formulation: list[dict]      # Final supplement stack with rationale
    citations: list[dict]              # PubMed DOIs per recommendation
    confidence_scores: dict            # Per-supplement confidence
    warnings: list[str]                # Safety warnings
    audit_trail: list[dict]            # Full decision log


# ── LLM Setup (Ollama — local Mistral, zero API cost) ─────────────────────────
def get_llm(temperature: float = 0.1) -> Ollama:
    """Mistral-7B-Instruct 4-bit quantized — fits in 6GB VRAM (RTX 4050)."""
    return Ollama(
        model="mistral:latest",
        base_url="http://localhost:11434",
        temperature=temperature,
        num_ctx=4096,
    )


# ── Vector Store (ChromaDB — local, persistent) ────────────────────────────────
def get_vector_store() -> Chroma:
    """Local ChromaDB with PubMed nutrition paper embeddings."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Free, runs on CPU
        model_kwargs={"device": "cpu"},
    )
    return Chroma(
        persist_directory="./vector_store/chroma",
        embedding_function=embeddings,
        collection_name="pubmed_nutrition",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 1: Research Agent
# Retrieves PubMed evidence for each deficiency using RAG
# ═══════════════════════════════════════════════════════════════════════════════
def research_agent(state: FormulationState) -> dict:
    logger.info("🔬 ResearchAgent: Retrieving PubMed evidence...")

    llm = get_llm(temperature=0.1)
    vector_store = get_vector_store()
    research_findings = []

    for deficiency in state["deficiencies"]:
        if deficiency["probability"] < 0.30:
            continue

        deficiency_name = deficiency["deficiency"].replace("_", " ")

        # RAG query
        query = f"treatment supplementation {deficiency_name} clinical trial evidence dosage"
        docs: list[Document] = vector_store.similarity_search(query, k=4)

        context = "\n\n".join([
            f"[Source {i+1}] PMID:{doc.metadata.get('pmid', 'unknown')}\n{doc.page_content}"
            for i, doc in enumerate(docs)
        ])

        prompt = PromptTemplate.from_template("""
You are a clinical nutritionist reviewing research evidence.

Patient has probable {deficiency} (probability: {probability:.0%}).

Research papers:
{context}

Based on this evidence, provide a structured JSON response:
{{
  "supplement_name": "specific supplement form (e.g., Vitamin D3 not just Vitamin D)",
  "evidence_summary": "2-3 sentences on what research shows",
  "recommended_dose_range": "e.g., 1000-2000 IU/day",
  "evidence_level": "strong/moderate/weak",
  "key_finding": "most important clinical finding in one sentence",
  "pmids": ["list of PMID numbers from sources"]
}}

Respond with ONLY the JSON, no other text.
""")

        try:
            response = llm.invoke(prompt.format(
                deficiency=deficiency_name,
                probability=deficiency["probability"],
                context=context,
            ))

            # Parse JSON response
            clean = response.strip().strip("```json").strip("```")
            finding = json.loads(clean)
            finding["deficiency"] = deficiency["deficiency"]
            finding["deficiency_probability"] = deficiency["probability"]
            finding["source_docs"] = [d.metadata for d in docs]
            research_findings.append(finding)

        except Exception as e:
            logger.warning(f"Research agent error for {deficiency_name}: {e}")
            # Fallback: minimal finding
            research_findings.append({
                "deficiency": deficiency["deficiency"],
                "supplement_name": deficiency_name.replace("_deficient", "").title(),
                "evidence_summary": "Research retrieval failed. Using conservative defaults.",
                "recommended_dose_range": "consult standard guidelines",
                "evidence_level": "weak",
                "key_finding": "Manual review recommended",
                "pmids": [],
            })

    logger.info(f"ResearchAgent: Found evidence for {len(research_findings)} deficiencies")
    return {"research_findings": research_findings}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 2: Dosing Agent
# Calculates personalized doses based on severity + patient context
# ═══════════════════════════════════════════════════════════════════════════════
def dosing_agent(state: FormulationState) -> dict:
    logger.info("💊 DosingAgent: Calculating personalized doses...")

    llm = get_llm(temperature=0.0)  # Deterministic for dosing
    dosing_adjustments = {}
    draft_formulation = []

    patient = state["patient_context"]
    age = patient.get("age", 35)
    sex = patient.get("sex", "unknown")
    weight_kg = patient.get("weight_kg", 70)
    conditions = patient.get("conditions", [])

    for finding in state["research_findings"]:
        deficiency_name = finding["deficiency"].replace("_", " ")
        severity = next(
            (d["severity"] for d in state["deficiencies"]
             if d["deficiency"] == finding["deficiency"]),
            "mild"
        )

        prompt = PromptTemplate.from_template("""
You are a clinical pharmacist calculating supplement doses.

Patient profile:
- Age: {age} years
- Sex: {sex}
- Weight: {weight_kg} kg
- Medical conditions: {conditions}
- Deficiency: {deficiency} ({severity} severity)
- Suggested range from research: {dose_range}

Calculate the optimal personalized dose considering:
1. Severity level (mild/moderate/severe)
2. Age-specific adjustments
3. Conditions that affect absorption (e.g., diabetes affects B12)
4. Therapeutic vs maintenance dosing

Respond ONLY with JSON:
{{
  "supplement": "{supplement_name}",
  "recommended_dose": "specific dose with unit",
  "frequency": "e.g., once daily with meals",
  "duration": "e.g., 3 months then reassess",
  "form": "preferred form for bioavailability (e.g., methylcobalamin not cyanocobalamin)",
  "timing": "best time to take (morning/evening/with food)",
  "dose_rationale": "one sentence explaining dose choice",
  "retest_in": "weeks until follow-up blood test"
}}
""")

        try:
            response = llm.invoke(prompt.format(
                age=age, sex=sex, weight_kg=weight_kg,
                conditions=", ".join(conditions) or "none",
                deficiency=deficiency_name, severity=severity,
                dose_range=finding.get("recommended_dose_range", "standard dosing"),
                supplement_name=finding.get("supplement_name", ""),
            ))

            clean = response.strip().strip("```json").strip("```")
            dosing = json.loads(clean)
            dosing["deficiency"] = finding["deficiency"]
            dosing["evidence_level"] = finding.get("evidence_level", "weak")
            dosing_adjustments[finding["deficiency"]] = dosing
            draft_formulation.append(dosing)

        except Exception as e:
            logger.warning(f"Dosing error for {deficiency_name}: {e}")

    logger.info(f"DosingAgent: Calculated doses for {len(draft_formulation)} supplements")
    return {"draft_formulation": draft_formulation, "dosing_adjustments": dosing_adjustments}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 3: Safety Agent
# Checks Neo4j knowledge graph for drug-supplement interactions
# ═══════════════════════════════════════════════════════════════════════════════
def safety_agent(state: FormulationState) -> dict:
    logger.info("🛡️ SafetyAgent: Checking drug-supplement interactions...")

    safety_flags = []
    current_meds = state["patient_context"].get("medications", [])
    conditions = state["patient_context"].get("conditions", [])

    # ── Neo4j interaction check ────────────────────────────────────────────────
    try:
        driver = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "nutrigenix123")
        )

        with driver.session() as session:
            for supplement_info in state["draft_formulation"]:
                supplement = supplement_info.get("supplement", "")

                # Query: Does this supplement interact with any current medication?
                if current_meds:
                    result = session.run("""
                        MATCH (s:Supplement {name: $supp})-[r:INTERACTS_WITH]->(d:Drug)
                        WHERE d.name IN $meds
                        RETURN s.name, d.name, r.severity, r.description
                        LIMIT 5
                    """, supp=supplement, meds=current_meds)

                    for record in result:
                        safety_flags.append({
                            "type": "drug_supplement_interaction",
                            "supplement": record["s.name"],
                            "drug": record["d.name"],
                            "severity": record["r.severity"],
                            "description": record["r.description"],
                            "action": "review_with_physician" if record["r.severity"] == "major" else "monitor",
                        })

                # Query: Contraindications with patient conditions
                if conditions:
                    result = session.run("""
                        MATCH (s:Supplement {name: $supp})-[r:CONTRAINDICATED_IN]->(c:Condition)
                        WHERE c.name IN $conditions
                        RETURN s.name, c.name, r.reason
                        LIMIT 3
                    """, supp=supplement, conditions=conditions)

                    for record in result:
                        safety_flags.append({
                            "type": "contraindication",
                            "supplement": record["s.name"],
                            "condition": record["c.name"],
                            "reason": record["r.reason"],
                            "action": "contraindicated_remove",
                        })

        driver.close()

    except Exception as e:
        logger.warning(f"Neo4j connection failed: {e}. Skipping graph-based safety check.")
        safety_flags.append({
            "type": "system_warning",
            "message": "Drug interaction database unavailable. Manual review advised.",
            "action": "manual_review",
        })

    # ── Rule-based safety checks (no Neo4j needed) ────────────────────────────
    supplements_in_plan = [s.get("supplement", "").lower() for s in state["draft_formulation"]]

    # Check for known problematic combinations
    KNOWN_CONFLICTS = [
        (["iron", "calcium"], "Iron and calcium compete for absorption. Take 2 hours apart."),
        (["vitamin d", "vitamin a"], "High-dose Vitamin A + D can cause toxicity. Monitor levels."),
        (["magnesium", "zinc"], "High-dose zinc inhibits magnesium absorption. Separate doses."),
        (["vitamin e", "fish oil"], "Both have anticoagulant effects. Monitor bleeding risk."),
    ]

    for conflicting_pair, message in KNOWN_CONFLICTS:
        if all(any(c in s for s in supplements_in_plan) for c in conflicting_pair):
            safety_flags.append({
                "type": "supplement_combination_warning",
                "supplements": conflicting_pair,
                "message": message,
                "action": "adjust_timing",
            })

    # Warfarin interactions (common)
    if "warfarin" in [m.lower() for m in current_meds]:
        warfarin_risk = ["vitamin k", "fish oil", "vitamin e", "coq10", "ginkgo"]
        for supp in supplements_in_plan:
            if any(r in supp for r in warfarin_risk):
                safety_flags.append({
                    "type": "anticoagulant_interaction",
                    "supplement": supp,
                    "drug": "warfarin",
                    "severity": "major",
                    "action": "physician_review_required",
                })

    logger.info(f"SafetyAgent: Found {len(safety_flags)} safety flags")
    return {"safety_flags": safety_flags}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 4: Formulation Agent
# Assembles the final supplement stack, removing contraindicated items
# ═══════════════════════════════════════════════════════════════════════════════
def formulation_agent(state: FormulationState) -> dict:
    logger.info("🧪 FormulationAgent: Assembling final formulation...")

    llm = get_llm(temperature=0.1)

    # Remove contraindicated supplements
    contraindicated = {
        flag["supplement"].lower()
        for flag in state["safety_flags"]
        if flag.get("action") == "contraindicated_remove"
    }

    filtered_formulation = [
        s for s in state["draft_formulation"]
        if s.get("supplement", "").lower() not in contraindicated
    ]

    # Add safety notes to flagged supplements
    for supplement in filtered_formulation:
        supp_name = supplement.get("supplement", "").lower()
        relevant_flags = [
            f for f in state["safety_flags"]
            if supp_name in f.get("supplement", "").lower()
        ]
        if relevant_flags:
            supplement["safety_notes"] = [f["message"] for f in relevant_flags if "message" in f]
            supplement["interactions"] = relevant_flags

    # Get LLM to write final clinical rationale for the complete stack
    if filtered_formulation:
        stack_summary = json.dumps(filtered_formulation, indent=2)
        deficiency_summary = json.dumps(
            [d for d in state["deficiencies"] if d["probability"] > 0.3],
            indent=2
        )

        prompt = f"""You are writing a clinical supplement formulation summary for a practitioner.

Patient deficiencies identified:
{deficiency_summary}

Proposed supplement stack:
{stack_summary}

Write a brief clinical narrative (3-4 sentences) explaining:
1. The primary deficiencies being addressed
2. Why this specific combination was chosen
3. Expected outcomes with timeline
4. Key monitoring parameters

Be concise and clinical in tone."""

        try:
            clinical_narrative = llm.invoke(prompt)
        except Exception:
            clinical_narrative = "Personalized supplement formulation based on biomarker analysis."
    else:
        clinical_narrative = "No supplements recommended after safety review. Consult physician."

    final_formulation = {
        "supplements": filtered_formulation,
        "removed_due_to_contraindication": list(contraindicated),
        "clinical_narrative": clinical_narrative,
        "total_supplements": len(filtered_formulation),
    }

    logger.info(f"FormulationAgent: Final stack has {len(filtered_formulation)} supplements")
    return {"final_formulation": filtered_formulation}


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT 5: Audit Agent
# Generates citations, confidence scores, and complete audit trail
# ═══════════════════════════════════════════════════════════════════════════════
def audit_agent(state: FormulationState) -> dict:
    logger.info("📋 AuditAgent: Building citations and audit trail...")

    citations = []
    confidence_scores = {}
    audit_trail = []

    for supplement in state["final_formulation"]:
        supp_name = supplement.get("supplement", "")
        deficiency = supplement.get("deficiency", "")

        # Find supporting research
        supporting_research = next(
            (f for f in state["research_findings"] if f.get("deficiency") == deficiency),
            {}
        )

        # Build citation entry
        pmids = supporting_research.get("pmids", [])
        citation = {
            "supplement": supp_name,
            "deficiency": deficiency,
            "pmids": pmids,
            "evidence_level": supporting_research.get("evidence_level", "unknown"),
            "key_finding": supporting_research.get("key_finding", ""),
            "pubmed_links": [f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" for pmid in pmids],
        }
        citations.append(citation)

        # Confidence score: combines deficiency probability + evidence level
        deficiency_prob = next(
            (d["probability"] for d in state["deficiencies"] if d["deficiency"] == deficiency),
            0.5
        )
        evidence_weights = {"strong": 1.0, "moderate": 0.7, "weak": 0.4, "unknown": 0.3}
        evidence_weight = evidence_weights.get(supporting_research.get("evidence_level", "unknown"), 0.3)
        confidence_scores[supp_name] = round(deficiency_prob * evidence_weight, 3)

        # Audit trail entry
        audit_trail.append({
            "supplement": supp_name,
            "decision_chain": [
                f"Biomarker analysis → {deficiency.replace('_', ' ')} detected (prob: {deficiency_prob:.0%})",
                f"PubMed RAG retrieved {len(supporting_research.get('source_docs', []))} papers",
                f"Evidence level: {supporting_research.get('evidence_level', 'unknown')}",
                f"Dose calculated for patient profile",
                f"Safety check: {len([f for f in state['safety_flags'] if supp_name.lower() in f.get('supplement', '').lower()])} flags",
                f"Final confidence: {confidence_scores[supp_name]:.0%}",
            ],
            "citations": pmids,
        })

    # Global warnings
    major_flags = [f for f in state["safety_flags"] if f.get("severity") == "major"]
    warnings = [f.get("description", f.get("message", "")) for f in major_flags]
    if state.get("patient_context", {}).get("medications"):
        warnings.append(
            "Patient is on prescription medications. Review all supplements with prescribing physician before starting."
        )

    logger.info(f"AuditAgent: Generated {len(citations)} citations, {len(audit_trail)} audit entries")
    return {
        "citations": citations,
        "confidence_scores": confidence_scores,
        "audit_trail": audit_trail,
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Build and compile the LangGraph pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def build_formulation_pipeline() -> StateGraph:
    """
    Constructs the 5-agent LangGraph pipeline.
    Agents run sequentially: Research → Dosing → Safety → Formulation → Audit
    """
    workflow = StateGraph(FormulationState)

    # Add all agents as nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("dosing", dosing_agent)
    workflow.add_node("safety", safety_agent)
    workflow.add_node("formulation", formulation_agent)
    workflow.add_node("audit", audit_agent)

    # Define execution order (sequential pipeline)
    workflow.set_entry_point("research")
    workflow.add_edge("research", "dosing")
    workflow.add_edge("dosing", "safety")
    workflow.add_edge("safety", "formulation")
    workflow.add_edge("formulation", "audit")
    workflow.add_edge("audit", END)

    return workflow.compile()


def run_formulation(
    patient_id: str,
    biomarkers: dict,
    deficiencies: list[dict],
    patient_context: dict,
) -> FormulationState:
    """
    Main entry point. Runs the full 5-agent pipeline.

    Args:
        patient_id: Unique patient identifier
        biomarkers: Raw biomarker dict from Layer 1
        deficiencies: Deficiency predictions from Layer 2
        patient_context: {age, sex, weight_kg, conditions, medications}

    Returns:
        Complete FormulationState with final_formulation, citations, audit_trail
    """
    pipeline = build_formulation_pipeline()

    initial_state: FormulationState = {
        "patient_id": patient_id,
        "biomarkers": biomarkers,
        "deficiencies": deficiencies,
        "patient_context": patient_context,
        "research_findings": [],
        "safety_flags": [],
        "draft_formulation": [],
        "dosing_adjustments": {},
        "final_formulation": [],
        "citations": [],
        "confidence_scores": {},
        "warnings": [],
        "audit_trail": [],
    }

    logger.info(f"Running formulation pipeline for patient {patient_id}...")
    result = pipeline.invoke(initial_state)
    logger.info("Pipeline complete.")

    return result
