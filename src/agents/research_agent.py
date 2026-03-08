from Bio import Entrez


class ResearchAgent:

    def __init__(self):
        Entrez.email = "research@nutrigenix.ai"

    def search_pubmed(self, query, max_results=3):

        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results
        )

        record = Entrez.read(handle)
        ids = record["IdList"]

        papers = []

        for pid in ids:

            summary = Entrez.esummary(
                db="pubmed",
                id=pid
            )

            data = Entrez.read(summary)

            title = data[0]["Title"]

            papers.append({
                "pmid": pid,
                "title": title
            })

        return papers