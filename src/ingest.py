"""
Document ingestion pipeline for the RAG Document Q&A Bot.

Reads PDFs from the data folder, extracts text page by page,
splits text into overlapping chunks, creates embeddings,
and stores everything in persistent ChromaDB.
"""

from pathlib import Path
import os

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# Load .env settings
load_dotenv()

# Project settings
DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "document_qa"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks."""

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def extract_pdf_chunks(pdf_path):
    """Extract text from every page of a PDF and create chunks."""

    reader = PdfReader(str(pdf_path))
    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        page_chunks = chunk_text(text)

        for chunk_number, chunk in enumerate(page_chunks):
            chunks.append(
                {
                    "text": chunk,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_number,
                    },
                }
            )

    return chunks


def main():
    """Index all PDF documents in the data folder."""

    print()
    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    if not DATA_DIR.exists():
        print()
        print("ERROR: data folder was not found.")
        raise SystemExit(1)

    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print()
        print("ERROR: No PDF files found in data folder.")
        raise SystemExit(1)

    print()
    print("PDF files found:")

    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    print()
    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Connecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    # Recreate collection so old/incorrect indexing is removed.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("Old collection removed.")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    all_chunks = []

    print()
    print("Extracting documents...")

    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}")

        pdf_chunks = extract_pdf_chunks(pdf_path)

        print(f"  Chunks created: {len(pdf_chunks)}")

        all_chunks.extend(pdf_chunks)

    if not all_chunks:
        print()
        print("ERROR: No text could be extracted from the PDFs.")
        raise SystemExit(1)

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print("Creating embeddings...")

    texts = [item["text"] for item in all_chunks]

    embeddings = embedding_model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    ids = [
        f"chunk_{index}"
        for index in range(len(all_chunks))
    ]

    metadatas = [
        item["metadata"]
        for item in all_chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    print()
    print(f"Indexed chunks: {collection.count()}")

    print()
    print("Indexed documents:")

    sources = sorted(
        set(
            metadata["source"]
            for metadata in metadatas
        )
    )

    for source in sources:
        count = sum(
            1
            for metadata in metadatas
            if metadata["source"] == source
        )

        print(f"  - {source}: {count} chunks")

    print()


if __name__ == "__main__":
    main()