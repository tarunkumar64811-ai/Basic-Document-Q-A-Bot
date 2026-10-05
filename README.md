\# RAG Document Q\&A Bot



A Retrieval-Augmented Generation (RAG) based document question-answering system.



The application retrieves relevant information from PDF documents using semantic search and generates grounded answers using a local Ollama LLM.



\## Features



\- PDF document ingestion

\- Page-level text extraction

\- Overlapping text chunking

\- Sentence Transformer embeddings

\- Persistent ChromaDB vector database

\- Top-k semantic retrieval

\- Local LLM answer generation using Ollama

\- Source filename and page citations

\- Grounded answers based only on retrieved document context

\- Interactive command-line interface



\## Technologies Used



\- Python

\- ChromaDB

\- Sentence Transformers

\- PyPDF

\- Ollama

\- Llama 3.2

\- python-dotenv



\## Documents



The system currently supports PDF documents placed inside the `data/` directory.



Example documents:



\- Python (Printed) Notes.pdf

\- CLOUD COMPUTING DIGITAL NOTES (R18A0523).pdf



\## Project Structure



```text

Ai intern/

│

├── data/

│   ├── Python (Printed) Notes.pdf

│   └── CLOUD COMPUTING DIGITAL NOTES (R18A0523).pdf

│

├── src/

│   ├── ingest.py

│   └── main.py

│

├── chroma\_db/

├── .env

├── .gitignore

├── README.md

└── requirements.txt

