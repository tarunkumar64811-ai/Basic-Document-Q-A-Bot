"""
Interactive RAG Document Q&A Bot.

Question -> query embedding -> similarity search -> retrieved chunks
-> LLM -> grounded answer + source citations
"""

from pathlib import Path
import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "document_qa"

TOP_K = int(os.getenv("TOP_K", "4"))
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print("No indexed collection was found.")
        print("First run: python src/ingest.py")
        raise SystemExit(1)

    if collection.count() == 0:
        print("The vector database is empty.")
        print("First run: python src/ingest.py")
        raise SystemExit(1)

    return collection


def retrieve(question, embedding_model, collection):
    """Retrieve the most similar chunks for the question."""
    # A single user query is embedded here for retrieval.
    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return list(zip(documents, metadatas, distances))


def build_context(retrieved_chunks):
    parts = []

    for index, (document, metadata, distance) in enumerate(
        retrieved_chunks, start=1
    ):
        parts.append(
            f"[SOURCE {index}]\n"
            f"File: {metadata['source']}\n"
            f"Page: {metadata['page']}\n"
            f"Content: {document}"
        )

    return "\n\n".join(parts)


def generate_answer(question, context):
    client = OpenAI()

    instructions = """You are a document question-answering assistant.

Rules:
1. Answer ONLY from the supplied document context.
2. If the context does not contain enough information to answer the question,
   say: "I couldn't find this information in the provided documents."
3. Do not use outside knowledge to fill gaps.
4. Keep the answer clear and concise.
5. At the end, list the source file and page number(s) used.
"""

    prompt = f"""DOCUMENT CONTEXT:

{context}

USER QUESTION:
{question}
"""

    response = client.responses.create(
        model=MODEL_NAME,
        instructions=instructions,
        input=prompt,
    )

    return response.output_text


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is missing.")
        print("Create a .env file and add your OpenAI API key.")
        raise SystemExit(1)

    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    collection = get_collection()

    print()
    print("=" * 60)
    print("RAG DOCUMENT Q&A BOT")
    print("=" * 60)
    print(f"Indexed chunks: {collection.count()}")
    print("Type 'exit' to quit.")
    print()

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        retrieved = retrieve(question, embedding_model, collection)
        context = build_context(retrieved)

        print("\nRetrieved sources:")
        for _, metadata, distance in retrieved:
            print(
                f"  - {metadata['source']} | "
                f"Page {metadata['page']} | "
                f"Distance {distance:.4f}"
            )

        print("\nGenerating answer...")
        answer = generate_answer(question, context)

        print("\nBot:")
        print(answer)
        print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    main()
