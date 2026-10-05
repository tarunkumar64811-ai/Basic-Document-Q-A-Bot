"""
Interactive RAG Document Q&A Bot
Uses ChromaDB for retrieval, SentenceTransformers for embeddings,
and Ollama for local LLM answer generation.
"""

from pathlib import Path
import os

import chromadb
import ollama
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Load .env settings
load_dotenv()

# Project settings
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "document_qa"
TOP_K = int(os.getenv("TOP_K", "4"))
MODEL_NAME = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def get_collection():
    """Open the existing ChromaDB collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print()
        print("ERROR: No indexed collection was found.")
        print("Run this first:")
        print("python src/ingest.py")
        raise SystemExit(1)

    if collection.count() == 0:
        print()
        print("ERROR: The vector database is empty.")
        print("Run this first:")
        print("python src/ingest.py")
        raise SystemExit(1)

    return collection


def retrieve(question, embedding_model, collection):
    """Find the most relevant document chunks."""
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
    """Build the context sent to Ollama."""
    parts = []

    for index, (document, metadata, distance) in enumerate(
        retrieved_chunks, start=1
    ):
        source = metadata.get("source", "Unknown")
        page = metadata.get("page", "Unknown")
        parts.append(
            f"[SOURCE {index}]\n"
            f"File: {source}\n"
            f"Page: {page}\n"
            f"Content:\n{document}"
        )

    return "\n\n".join(parts)


def generate_answer(question, context):
    """Generate a grounded answer using the local Ollama model."""

    system_prompt = """You are a document question-answering assistant.

Rules:
1. Answer only from the supplied DOCUMENT CONTEXT.
2. Do not use outside knowledge.
3. If the answer is not present in the context, say:
   "I couldn't find this information in the provided documents."
4. Keep the answer clear and concise.
5. At the end, write the source file name and page number(s) used.
"""

    user_prompt = f"""DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        options={
            "temperature": 0,
        },
    )

    return response["message"]["content"]


def main():
    """Start the interactive RAG application."""

    print()
    print("=" * 60)
    print("RAG DOCUMENT Q&A BOT")
    print("=" * 60)

    print()
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Connecting to vector database...")
    collection = get_collection()

    print()
    print(f"Indexed chunks: {collection.count()}")
    print(f"Local LLM: {MODEL_NAME}")
    print("Type 'exit' or 'quit' to stop.")
    print()

    while True:
        question = input("You: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            continue

        retrieved = retrieve(
            question,
            embedding_model,
            collection,
        )

        context = build_context(retrieved)

        print()
        print("Retrieved sources:")

        for _, metadata, distance in retrieved:
            source = metadata.get("source", "Unknown")
            page = metadata.get("page", "Unknown")

            print(
                f"  - {source} | "
                f"Page {page} | "
                f"Distance {distance:.4f}"
            )

        print()
        print("Generating answer...")

        try:
            answer = generate_answer(
                question,
                context,
            )

            print()
            print("Bot:")
            print(answer)

        except Exception as error:
            print()
            print("ERROR while generating the answer:")
            print(error)
            print()
            print("Check that Ollama is running and the model is installed.")
            print("Run: ollama list")

        print()
        print("-" * 60)
        print()


if __name__ == "__main__":
    main()
