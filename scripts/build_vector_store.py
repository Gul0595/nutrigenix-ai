"""
scripts/build_vector_store.py
==============================
Downloads nutrition-related PubMed abstracts and builds a
ChromaDB vector store for RAG. Runs once (~30-45 min).
Uses free NCBI E-utilities API (no key needed, key increases rate limit).
"""

import os
import time
import json
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from Bio import Entrez
from loguru import logger
from tqdm import tqdm


# ── Configuration ─────────────────────────────────────────────────────────────
Entrez.email = "nutrigenix@example.com"   # Required by NCBI
PUBMED_API_KEY = os.getenv("PUBMED_API_KEY", "")  # Optional: get free key at NCBI
if PUBMED_API_KEY:
    Entrez.api_key = PUBMED_API_KEY

CHROMA_DIR = "./vector_store/chroma"
COLLECTION_NAME = "pubmed_nutrition"
BATCH_SIZE = 100       # Articles per API request
MAX_ARTICLES = 10000   # Start with 10k, expand later

# Nutrition + supplement search queries targeting our deficiencies
SEARCH_QUERIES = [
    "vitamin D deficiency supplementation randomized trial",
    "vitamin B12 deficiency treatment methylcobalamin",
    "iron deficiency anemia supplementation",
    "magnesium supplementation clinical benefits",
    "zinc deficiency immune function supplementation",
    "folate folic acid supplementation pregnancy",
    "omega-3 fish oil cardiovascular clinical trial",
    "coenzyme Q10 supplementation heart disease",
    "vitamin C supplementation immune function",
    "calcium vitamin D bone health supplementation",
    "probiotic gut health randomized trial",
    "curcumin anti-inflammatory bioavailability",
    "ashwagandha adaptogen stress anxiety trial",
    "nutraceutical personalized nutrition biomarker",
    "micronutrient deficiency blood biomarker diagnosis",
]


def fetch_pubmed_abstracts(query: str, max_results: int = 500) -> list[dict]:
    """Fetch abstracts from PubMed using E-utilities API (free)."""
    articles = []

    try:
        # Search for PMIDs
        search_handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
            datetype="pdat",
            mindate="2015",  # Last 10 years
        )
        search_results = Entrez.read(search_handle)
        search_handle.close()
        pmids = search_results.get("IdList", [])

        if not pmids:
            return articles

        # Fetch abstracts in batches
        for i in range(0, len(pmids), BATCH_SIZE):
            batch = pmids[i:i + BATCH_SIZE]
            fetch_handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch),
                rettype="abstract",
                retmode="xml",
            )
            records = Entrez.read(fetch_handle)
            fetch_handle.close()

            for record in records.get("PubmedArticle", []):
                try:
                    article = record["MedlineCitation"]["Article"]
                    pmid = str(record["MedlineCitation"]["PMID"])

                    title = str(article.get("ArticleTitle", ""))
                    abstract_text = ""
                    if "Abstract" in article:
                        abstract_sections = article["Abstract"].get("AbstractText", [])
                        if isinstance(abstract_sections, list):
                            abstract_text = " ".join(str(s) for s in abstract_sections)
                        else:
                            abstract_text = str(abstract_sections)

                    if not abstract_text or len(abstract_text) < 100:
                        continue

                    # Extract publication year
                    pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                    year = str(pub_date.get("Year", "unknown"))

                    articles.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract_text,
                        "year": year,
                        "query": query,
                    })

                except Exception:
                    continue

            # Respect NCBI rate limits: 10 req/sec with key, 3/sec without
            time.sleep(0.15 if PUBMED_API_KEY else 0.35)

    except Exception as e:
        logger.error(f"PubMed fetch error for '{query}': {e}")

    return articles


def build_vector_store():
    """Main function to build ChromaDB from PubMed abstracts."""
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)

    # Load embedding model (runs on CPU, no GPU needed)
    logger.info("Loading sentence transformer embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64},
    )

    # Initialize ChromaDB
    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    seen_pmids = set()
    total_added = 0

    logger.info(f"Fetching PubMed abstracts for {len(SEARCH_QUERIES)} queries...")

    for query in tqdm(SEARCH_QUERIES, desc="PubMed queries"):
        articles = fetch_pubmed_abstracts(query, max_results=500)
        logger.info(f"  '{query[:50]}...' → {len(articles)} articles")

        # Build LangChain Document objects
        docs = []
        for art in articles:
            if art["pmid"] in seen_pmids:
                continue
            seen_pmids.add(art["pmid"])

            # Combine title + abstract for richer embedding
            content = f"Title: {art['title']}\n\nAbstract: {art['abstract']}"

            docs.append(Document(
                page_content=content,
                metadata={
                    "pmid": art["pmid"],
                    "title": art["title"],
                    "year": art["year"],
                    "source": "pubmed",
                    "query_topic": query.split()[0],  # Topic tag for filtering
                }
            ))

        if docs:
            # Add to ChromaDB in batches (avoid memory issues)
            batch_size = 100
            for i in range(0, len(docs), batch_size):
                vector_store.add_documents(docs[i:i + batch_size])

            total_added += len(docs)
            logger.info(f"  Added {len(docs)} documents (total: {total_added})")

        time.sleep(1)  # Brief pause between queries

    # Persist
    vector_store.persist()
    logger.info(f"\n✅ Vector store built: {total_added} unique PubMed abstracts")
    logger.info(f"   Saved to: {CHROMA_DIR}")
    logger.info(f"   Collection: {COLLECTION_NAME}")

    # Quick test
    logger.info("\nRunning test query...")
    results = vector_store.similarity_search("vitamin D deficiency treatment dose", k=3)
    for r in results:
        logger.info(f"  PMID {r.metadata['pmid']}: {r.metadata['title'][:80]}...")


if __name__ == "__main__":
    build_vector_store()
