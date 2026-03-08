"""
scripts/build_knowledge_graph.py
==================================
Builds the Neo4j supplement-drug-condition interaction graph
from free data sources: DrugBank open data + NIH ODS + curated rules.
"""

from neo4j import GraphDatabase
from loguru import logger


KNOWN_INTERACTIONS = [
    # (supplement, drug, severity, description)
    ("Vitamin K", "Warfarin", "major", "Vitamin K reduces anticoagulant effect of warfarin. Monitor INR closely."),
    ("Vitamin E", "Warfarin", "moderate", "High-dose Vitamin E may enhance anticoagulant effect."),
    ("Fish Oil", "Warfarin", "moderate", "Omega-3 may increase bleeding risk when combined with warfarin."),
    ("St John Wort", "Warfarin", "major", "Significantly reduces warfarin plasma levels."),
    ("Calcium", "Levothyroxine", "moderate", "Calcium reduces thyroid hormone absorption. Separate by 4 hours."),
    ("Iron", "Levothyroxine", "moderate", "Iron chelates levothyroxine. Separate by 2-4 hours."),
    ("Magnesium", "Antibiotics", "moderate", "Magnesium reduces absorption of fluoroquinolone antibiotics."),
    ("Zinc", "ACE Inhibitors", "minor", "Zinc may reduce ACE inhibitor efficacy."),
    ("Potassium", "ACE Inhibitors", "moderate", "Risk of hyperkalemia. Monitor potassium levels."),
    ("Vitamin B6", "Levodopa", "major", "High B6 reduces effectiveness of levodopa in Parkinson's."),
    ("CoQ10", "Warfarin", "moderate", "May reduce anticoagulant effect of warfarin."),
    ("Ginkgo", "Warfarin", "major", "Significantly increases bleeding risk."),
    ("Garlic", "Warfarin", "moderate", "May increase anticoagulant effect."),
    ("Melatonin", "Warfarin", "moderate", "May increase anticoagulant effect."),
    ("Vitamin C", "Warfarin", "minor", "Very high doses may increase warfarin effect."),
]

CONTRAINDICATIONS = [
    # (supplement, condition, reason)
    ("Iron", "Hemochromatosis", "Iron overload condition — supplemental iron contraindicated."),
    ("Vitamin A", "Pregnancy", "High doses teratogenic. Avoid >3000 IU preformed vitamin A."),
    ("Kava", "Liver Disease", "Hepatotoxic — contraindicated with any liver condition."),
    ("High-dose Zinc", "Copper Deficiency", "Long-term high zinc causes copper deficiency."),
    ("Calcium", "Hypercalcemia", "Do not supplement calcium if already hypercalcemic."),
    ("Vitamin D", "Hypercalcemia", "Vitamin D raises calcium — contraindicated."),
    ("Potassium", "Kidney Disease", "Impaired excretion — hyperkalemia risk."),
    ("High-dose Magnesium", "Kidney Disease", "Kidneys cannot excrete excess magnesium."),
]

ABSORPTION_SYNERGIES = [
    # (supplement1, supplement2, benefit)
    ("Vitamin D3", "Vitamin K2", "K2 directs calcium to bones; take together for best effect"),
    ("Iron", "Vitamin C", "Vitamin C enhances non-heme iron absorption by 2-3x"),
    ("Curcumin", "Black Pepper Extract", "Piperine increases curcumin bioavailability by 2000%"),
    ("Omega-3", "Vitamin E", "Vitamin E prevents oxidation of omega-3 fats"),
    ("Magnesium", "Vitamin B6", "B6 increases intracellular magnesium accumulation"),
    ("Zinc", "Vitamin A", "Zinc required for vitamin A transport and metabolism"),
]


def build_graph(uri: str = "bolt://localhost:7687",
                user: str = "neo4j",
                password: str = "nutrigenix123"):

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        # ── Clear existing data ────────────────────────────────────────────────
        session.run("MATCH (n) DETACH DELETE n")
        logger.info("Cleared existing graph data")

        # ── Create constraints ─────────────────────────────────────────────────
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Supplement) REQUIRE s.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE")
        session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Condition) REQUIRE c.name IS UNIQUE")

        # ── Load drug interactions ────────────────────────────────────────────
        logger.info("Loading supplement-drug interactions...")
        for supp, drug, severity, desc in KNOWN_INTERACTIONS:
            session.run("""
                MERGE (s:Supplement {name: $supp})
                MERGE (d:Drug {name: $drug})
                MERGE (s)-[r:INTERACTS_WITH {severity: $severity, description: $desc}]->(d)
            """, supp=supp, drug=drug, severity=severity, desc=desc)

        # ── Load contraindications ────────────────────────────────────────────
        logger.info("Loading contraindications...")
        for supp, condition, reason in CONTRAINDICATIONS:
            session.run("""
                MERGE (s:Supplement {name: $supp})
                MERGE (c:Condition {name: $condition})
                MERGE (s)-[r:CONTRAINDICATED_IN {reason: $reason}]->(c)
            """, supp=supp, condition=condition, reason=reason)

        # ── Load absorption synergies ─────────────────────────────────────────
        logger.info("Loading absorption synergies...")
        for supp1, supp2, benefit in ABSORPTION_SYNERGIES:
            session.run("""
                MERGE (s1:Supplement {name: $s1})
                MERGE (s2:Supplement {name: $s2})
                MERGE (s1)-[r:SYNERGIZES_WITH {benefit: $benefit}]->(s2)
            """, s1=supp1, s2=supp2, benefit=benefit)

        # ── Stats ─────────────────────────────────────────────────────────────
        result = session.run("MATCH (n) RETURN labels(n)[0] as type, count(n) as count")
        for record in result:
            logger.info(f"  {record['type']}: {record['count']} nodes")

        result = session.run("MATCH ()-[r]->() RETURN type(r) as type, count(r) as count")
        for record in result:
            logger.info(f"  {record['type']}: {record['count']} relationships")

    driver.close()
    logger.info("✅ Knowledge graph built successfully")
    logger.info("   View at: http://localhost:7474 (user: neo4j / pass: nutrigenix123)")


if __name__ == "__main__":
    build_graph()
