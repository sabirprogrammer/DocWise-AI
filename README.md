# 📚 AI Document Assistant

A simple Streamlit RAG-style document assistant that lets you upload documents, build a reusable local FAISS vector index, and ask questions grounded in the retrieved document text.

## Features

- Upload multiple **PDF, DOCX, TXT and MD** files
- Separate extraction functions for every supported format
- PDF page numbers are preserved
- Overlapping text chunking
- Filename and page metadata stored with every chunk
- Sentence Transformers embeddings using `all-MiniLM-L6-v2`
- FAISS semantic vector search
- Simple keyword search
- Hybrid ranking that combines semantic and keyword scores
- Groq-generated answers restricted to retrieved context
- Retrieved source chunks shown below each answer
- Streamlit session state prevents rebuilding embeddings for every question
- Streamlit resource caching reuses the embedding model
- Google Drive shared file-link loading
- Groq API key stored in Streamlit Secrets, never hardcoded

## Project structure

```text
AI-Document-Assistant/
├── app.py
├── requirements.txt
└── README.md
```

## How the pipeline works

```text
Documents
   ↓
Text extraction
   ↓
Overlapping chunks + metadata
   ↓
Sentence Transformer embeddings
   ↓
FAISS vector index
   ↓
Question
   ├── Semantic search
   └── Keyword search
          ↓
      Hybrid ranking
          ↓
Retrieved context
          ↓
        Groq
          ↓
Answer + sources
```

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Then run:

```bash
streamlit run app.py
```

## Streamlit Community Cloud

1. Create a new Streamlit app from this repository.
2. Set the main file to `app.py`.
3. Open the app's **Secrets** settings.
4. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

5. Deploy the app.

Do not commit your real API key to GitHub.

## Google Drive

Paste a publicly shared Google Drive **file** link in the sidebar. The file should be shared as **Anyone with the link** and must be PDF, DOCX, TXT or MD.

Native Google Docs links are exported as DOCX automatically.

### Folder note

Google Drive does not provide a reliable unauthenticated API for listing every file inside a public folder. Reliable folder ingestion requires Google Drive API/OAuth credentials. To keep this beginner-friendly project to only `app.py`, `requirements.txt`, and `README.md`, the included implementation accepts individual public Drive file links and clearly reports the limitation for folder links.

## Main functions

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`
- `create_chunks()`
- `build_vector_index()`
- `semantic_search()`
- `keyword_search()`
- `hybrid_search()`
- `ask_groq()`
- `files_signature()`

## Why embeddings are not recreated for every question

The processed chunks, embeddings and FAISS index are stored in `st.session_state`. A SHA-256 signature is calculated from the selected documents. If the same files remain selected, asking another question reuses the existing FAISS index.

The Sentence Transformers model itself is loaded with `@st.cache_resource`.

## Models

Embeddings:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Groq model:

```text
llama-3.3-70b-versatile
```

## Important limitations

- Scanned/image-only PDFs need OCR, which is not included.
- DOCX page metadata is unavailable with normal text extraction.
- The FAISS index lives in the current Streamlit session.
- Google Drive folder ingestion needs OAuth/API support.
- Large document collections may need persistent vector storage.

## Security

Keep `GROQ_API_KEY` only in Streamlit Secrets or local `.streamlit/secrets.toml`. Never place the real key inside `app.py`, `README.md`, or a public GitHub repository.
