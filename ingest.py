import os
from pathlib import Path
from dotenv import load_dotenv
import voyageai
from supabase import create_client
from pypdf import PdfReader

load_dotenv()

voyage   = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def ingest_pdf(pdf_path: str, source_title: str, source_url: str = ""):
    print(f"Processing: {source_title}...")

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "

    chunks = chunk_text(full_text)
    print(f"  → {len(chunks)} chunks created")

    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = [c for c in chunks[i:i + batch_size] if len(c.strip()) > 50]
        if not batch:
            continue

        result = voyage.embed(batch, model="voyage-3-lite", input_type="document")

        for chunk, embedding in zip(batch, result.embeddings):
            supabase.table("documents").insert({
                "content": chunk,
                "embedding": embedding,
                "source_title": source_title,
                "source_url": source_url
            }).execute()

        print(f"  → Stored {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

    print(f"  ✓ Done: {source_title}")


# ── ADD YOUR PDF SOURCES HERE ──────────────────────────────────────────
SOURCES = [
    {
        "path": "sources/ACOG_postpartum1.pdf",
        "title": "ACOG Optimizing Postpartum Care",
        "url": "https://www.acog.org",
    },
    {
        "path": "sources/content.pdf",
        "title": "Postpartum Care Clinical Guidelines",
        "url": "https://iris.who.int/server/api/core/bitstreams/73dec697-c033-449c-8323-1cd04a8d8f20/content",
    },
    {
        "path": "sources/NCBI_breastfeeding.pdf",
        "title": "NCBI Breastfeeding Guidelines",
        "url": "https://www.ncbi.nlm.nih.gov",
    },
    {
        "path": "sources/PubMed_1Yr.pdf",
        "title": "PubMed Postpartum Care 1 Year",
        "url": "https://pubmed.ncbi.nlm.nih.gov/37315166/",
    },
    {
        "path": "sources/PubMed_02.pdf",
        "title": "PubMed Postpartum Research",
        "url": "https://pubmed.ncbi.nlm.nih.gov/15330882/",
    },
    {
        "path": "sources/WHO_2.pdf",
        "title": "WHO Postpartum Care Guidelines",
        "url": "https://www.who.int",
    },
]

if __name__ == "__main__":
    print("Starting ingestion...")
    for source in SOURCES:
        if Path(source["path"]).exists():
            ingest_pdf(source["path"], source["title"], source["url"])
        else:
            print(f"WARNING: File not found — {source['path']}")
    print("\nDone. Your knowledge base is ready.")