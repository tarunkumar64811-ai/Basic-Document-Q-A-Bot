# Basic Document Q&A Bot using RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) application that lets a user ask questions about a collection of PDF documents. The system retrieves relevant document chunks, gives those chunks to an LLM, and returns a grounded answer with source information.

## 1. Architecture

```text
PDF documents
     |
     v
PDF text extraction
     |
     v
Text chunking + overlap
     |
     v
Sentence Transformer embeddings
     |
     v
Persistent ChromaDB
     |
     |  user question
     v
Query embedding
     |
     v
Similarity search (top-k)
     |
     v
Retrieved document chunks
     |
     v
OpenAI LLM
     |
     v
Grounded answer + source citations
```

## 2. Tech stack

- Python 3.11+
- pypdf - PDF text extraction
- sentence-transformers - local text embeddings
- ChromaDB - persistent vector database
- OpenAI Python SDK - LLM answer generation
- python-dotenv - environment variables

## 3. Chunking strategy

The first implementation uses fixed-size character chunks with overlap.

Default:
- Chunk size: 900 characters
- Overlap: 150 characters

Overlap helps preserve context when an important sentence crosses a chunk boundary.

## 4. Embedding model

`all-MiniLM-L6-v2` from Sentence Transformers is used locally for embeddings.

The document chunks are embedded in a batch during indexing.

## 5. Vector database

ChromaDB is used with a local persistent client. The vector database is stored in `./chroma_db`.

The indexing step and querying step are separate:
- `python src/ingest.py`
- `python src/main.py`

## 6. LLM

The application uses an OpenAI model through the Responses API.

The model name is configurable using `OPENAI_MODEL` in `.env`.

## 7. Setup

### Windows

Open the project folder in VS Code.

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```cmd
.venv\Scriptsctivate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and put your API key in `.env`.

Never commit `.env`.

## 8. Add documents

Put 4–5 meaningful PDFs in:

```text
data/
```

For the first test, one or two PDFs are enough.

For the final submission, use 4–5 documents that meet the assignment requirements.

## 9. Index documents

Run:

```powershell
python src/ingest.py
```

This:
1. Reads the PDFs.
2. Extracts page text.
3. Splits the text into chunks.
4. Creates embeddings in batches.
5. Stores chunks, metadata, and embeddings in ChromaDB.

## 10. Run the bot

```powershell
python src/main.py
```

Example:

```text
You: What is machine learning?

Bot:
...
Source: machine_learning.pdf, Page 3
```

## 11. Example queries

Use questions whose answers are actually present in your documents.

Also test an unanswerable question, such as a topic that is not covered by any document.

The bot is instructed not to use outside knowledge when the answer is absent from the retrieved context.

## 12. Known limitations

- Scanned/image-only PDFs need OCR.
- Fixed-size character chunking is simple rather than semantic.
- Retrieval quality depends on the selected embedding model and document quality.
- The final answer depends on the LLM.
- The first run downloads the embedding model.

## 13. Security

Never put an API key directly in source code.

Use `.env`, and keep `.env` in `.gitignore`.

## 14. Suggested project explanation for an interview

> "I built a basic RAG document question-answering system. First, I extract text from PDF documents and split it into overlapping chunks. I generate embeddings for those chunks and store them in a persistent ChromaDB vector database. When the user asks a question, I embed the query and perform similarity search to retrieve the most relevant chunks. I then pass those chunks as context to an LLM and instruct it to answer only from the retrieved context. Finally, the application displays the answer along with the source document and page information."

