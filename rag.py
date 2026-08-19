import os
from dotenv import load_dotenv
import anthropic
import voyageai
from supabase import create_client

load_dotenv()

claude   = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
voyage   = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])
supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_SERVICE_KEY"]
)

SYSTEM_PROMPT = """You are Bloom, a warm and knowledgeable postpartum companion.
You support new mothers through the physical and emotional challenges of the postpartum period.

Rules you always follow:
- Answer ONLY based on the source passages provided to you. Do not use outside knowledge.
- Be warm, empathetic, and non-judgmental — like a trusted friend who is also medically informed.
- If the provided sources don't contain enough information to answer, say: "I want to make sure I give you accurate information. For this specific question, I'd recommend speaking with your OB or midwife."
- NEVER diagnose or prescribe. Always recommend professional care for clinical concerns.
- End every response with a "Sources:" section listing the documents you drew from.
- Keep responses concise — under 300 words unless the question genuinely requires more detail."""


def get_embedding(text: str) -> list:
    result = voyage.embed([text], model="voyage-3-lite", input_type="query")
    return result.embeddings[0]


def search_documents(query: str, match_count: int = 5) -> list:
    query_embedding = get_embedding(query)
    result = supabase.rpc(
        "match_documents",
        {
            "query_embedding": query_embedding,
            "match_count": match_count
        }
    ).execute()
    return result.data


def build_context(passages: list) -> str:
    if not passages:
        return "No relevant source passages found."
    context = ""
    for i, passage in enumerate(passages, 1):
        context += f"\n[Source {i}: {passage['source_title']}]\n"
        context += passage['content']
        context += "\n"
    return context


def ask(question: str) -> dict:
    passages = search_documents(question)
    context  = build_context(passages)

    response = claude.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"""Here are the relevant source passages to answer the question:

{context}

---

Question: {question}

Please answer based on the sources above."""
            }
        ]
    )

    answer = response.content[0].text

    sources_used = [
        {"title": p["source_title"], "url": p.get("source_url", "")}
        for p in passages
        if p.get("similarity", 0) > 0.3
    ]

    return {"answer": answer, "sources": sources_used}