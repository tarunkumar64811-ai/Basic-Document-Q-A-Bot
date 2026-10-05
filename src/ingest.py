"""
Index PDF documents into a persistent Chroma vector database.

Pipeline:
PDF files -> text extraction -> chunks -> embeddings -> ChromaDB
"""

from pathlib import Path
import os

import chromadb
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()

DATA_DIR = Path("data")
CHROMA_DIR = Path("chroma_db")
COLLECTION_NAME = "document_qa"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def extract_pages(pdf_path: Path):
    """Return a list of (page_number, text) pairs."""
    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append((page_number, text))

    return pages


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping character chunks."""
    if overlap >= chunk_size:
        raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


def main():
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data/ folder.")
        print("Add at least one PDF and run this command again:")
        print("python src/ingest.py")
        return

    print(f"Found {len(pdf_files)} PDF file(s).")

    # Load the local embedding model once.
    print("Loading embedding model...")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    documents = []
    metadatas = []
    ids = []

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        for page_number, page_text in extract_pages(pdf_path):
            page_chunks = chunk_text(page_text)

            for chunk_index, chunk in enumerate(page_chunks):
                documents.append(chunk)
                metadatas.append(
                    {
                        "source": pdf_path.name,
                        "page": page_number,
                        "chunk": chunk_index,
                    }
                )
                ids.append(
                    f"{pdf_path.stem}-p{page_number}-c{chunk_index}"
                )

    if not documents:
        print("No extractable text was found in the PDFs.")
        print("If your PDF is scanned images, OCR will be needed.")
        return

    print(f"Created {len(documents)} chunks.")
    print("Creating embeddings in batches...")

    # One batched call for the document chunks.
    embeddings = embedding_model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # Re-indexing should replace old records instead of duplicating them.
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print()
    print("INDEXING COMPLETE")
    print(f"Stored chunks: {collection.count()}")
    print(f"Vector database: {CHROMA_DIR.resolve()}")
    print()
    print("Now run:")
    print("python src/main.py")


if __name__ == "__main__":
    main()
