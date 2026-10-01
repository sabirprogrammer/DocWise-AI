import streamlit as st
import io
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse

import requests
import faiss
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from groq import Groq


st.set_page_config(page_title="AI Document Assistant", page_icon="📚", layout="wide")

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    records = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            records.append({
                "text": text,
                "filename": filename,
                "page": page_number,
            })

    return records


def extract_docx(file_bytes, filename):
    document = Document(io.BytesIO(file_bytes))
    text = "\n".join(
        p.text for p in document.paragraphs if p.text.strip()
    ).strip()

    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }] if text else []


def extract_txt(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }] if text else []


def extract_md(file_bytes, filename):
    text = file_bytes.decode("utf-8", errors="ignore").strip()
    return [{
        "text": text,
        "filename": filename,
        "page": None,
    }] if text else []


def extract_file(file_bytes, filename):
    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_bytes, filename)
    if extension == ".docx":
        return extract_docx(file_bytes, filename)
    if extension == ".txt":
        return extract_txt(file_bytes, filename)
    if extension == ".md":
        return extract_md(file_bytes, filename)

    raise ValueError(f"Unsupported file type: {extension}")


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def create_chunks(records):
    chunks = []

    for record in records:
        for chunk_number, chunk_text in enumerate(
            split_text(record["text"]),
            start=1,
        ):
            chunks.append({
                "text": chunk_text,
                "filename": record["filename"],
                "page": record.get("page"),
                "chunk_number": chunk_number,
            })

    return chunks


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


def build_vector_index(chunks):
    model = load_embedding_model()
    texts = [chunk["text"] for chunk in chunks]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embeddings, index


def semantic_search(question, index, chunks, top_k=8):
    model = load_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    k = min(top_k, len(chunks))
    scores, indices = index.search(query_embedding, k)

    results = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results[int(idx)] = float(score)

    return results


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "what", "when", "where", "which", "who",
    "why", "with", "you", "your", "does", "do", "did", "can", "could"
}


def important_words(text):
    words = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    return {
        word for word in words
        if len(word) > 2 and word not in STOPWORDS
    }


def keyword_search(question, chunks):
    query_words = important_words(question)
    scores = {}

    if not query_words:
        return scores

    for idx, chunk in enumerate(chunks):
        chunk_words = important_words(chunk["text"])
        matches = query_words.intersection(chunk_words)

        if matches:
            scores[idx] = len(matches) / len(query_words)

    return scores


def hybrid_search(question, index, chunks, top_k=5):
    semantic = semantic_search(
        question,
        index,
        chunks,
        top_k=min(max(top_k * 4, 10), len(chunks)),
    )
    keyword = keyword_search(question, chunks)

    candidate_ids = set(semantic) | set(keyword)
    ranked = []

    for idx in candidate_ids:
        semantic_score = max(0.0, semantic.get(idx, 0.0))
        keyword_score = keyword.get(idx, 0.0)
        combined_score = (
            0.75 * semantic_score
            + 0.25 * keyword_score
        )

        result = dict(chunks[idx])
        result["semantic_score"] = semantic_score
        result["keyword_score"] = keyword_score
        result["score"] = combined_score
        ranked.append(result)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def get_drive_file_id(url):
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


def get_drive_folder_id(url):
    match = re.search(r"/folders/([a-zA-Z0-9_-]+)", url)
    return match.group(1) if match else None


def download_drive_file(url):
    parsed = urlparse(url)
    path = parsed.path

    doc_match = re.search(
        r"/document/d/([a-zA-Z0-9_-]+)",
        path,
    )

    if doc_match:
        file_id = doc_match.group(1)
        download_url = (
            f"https://docs.google.com/document/d/"
            f"{file_id}/export?format=docx"
        )

        response = requests.get(download_url, timeout=60)
        response.raise_for_status()

        return [(
            response.content,
            f"google_doc_{file_id}.docx",
        )]

    file_id = get_drive_file_id(url)
    if not file_id:
        raise ValueError(
            "Could not find a Google Drive file ID in this link."
        )

    download_url = (
        "https://drive.google.com/uc?"
        f"export=download&id={file_id}"
    )

    response = requests.get(download_url, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get(
        "Content-Type",
        "",
    ).lower()

    disposition = response.headers.get(
        "Content-Disposition",
        "",
    )

    if "text/html" in content_type:
        raise ValueError(
            "Drive returned an HTML page instead of the file. "
            "Make sure the file is shared as 'Anyone with the link'."
        )

    filename = None
    filename_match = re.search(
        r'filename="?([^";]+)"?',
        disposition,
    )

    if filename_match:
        filename = filename_match.group(1)

    if not filename:
        raise ValueError(
            "The Drive file downloaded, but its filename/type "
            "could not be detected. Use a direct shared file link "
            "or upload it locally."
        )

    return [(response.content, filename)]


def download_drive_folder(url):
    folder_id = get_drive_folder_id(url)

    if not folder_id:
        raise ValueError(
            "Could not find a Google Drive folder ID in this link."
        )

    raise ValueError(
        "Public Drive file links work directly. "
        "Reliable Drive folder loading requires Google Drive "
        "API/OAuth. Use individual shared file links for now."
    )


def load_drive_link(url):
    if get_drive_folder_id(url):
        return download_drive_folder(url)

    return download_drive_file(url)


def files_signature(files):
    digest = hashlib.sha256()

    for filename, file_bytes in sorted(
        files,
        key=lambda item: item[0],
    ):
        digest.update(
            filename.encode("utf-8", errors="ignore")
        )
        digest.update(file_bytes)

    return digest.hexdigest()


def process_documents(files):
    records = []
    file_info = []

    for filename, file_bytes in files:
        extracted = extract_file(
            file_bytes,
            filename,
        )
        records.extend(extracted)

        file_info.append({
            "filename": filename,
            "type": Path(filename).suffix
                .lower()
                .replace(".", "")
                .upper(),
            "pages_or_sections": len(extracted),
            "characters": sum(
                len(item["text"])
                for item in extracted
            ),
        })

    chunks = create_chunks(records)

    if not chunks:
        raise ValueError(
            "No readable text was found "
            "in the selected documents."
        )

    embeddings, index = build_vector_index(chunks)

    return {
        "records": records,
        "chunks": chunks,
        "embeddings": embeddings,
        "index": index,
        "file_info": file_info,
    }


def ask_groq(question, retrieved_chunks):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except Exception as exc:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets before asking questions."
        ) from exc

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):
        page = (
            f", page {item['page']}"
            if item.get("page")
            else ""
        )

        context_parts.append(
            f"[Source {number}: "
            f"{item['filename']}{page}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""You are an AI Document Assistant.

Answer the user's question ONLY from the supplied document context.
Do not use outside knowledge.
If the answer is not supported by the context, reply:
"I could not find that information in the provided documents."

Be clear and concise.

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}
"""

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer only from the provided "
                    "document context."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.1,
    )

    return response.choices[0].message.content


if "processed_signature" not in st.session_state:
    st.session_state.processed_signature = None

if "document_data" not in st.session_state:
    st.session_state.document_data = None

if "drive_files" not in st.session_state:
    st.session_state.drive_files = []

if "messages" not in st.session_state:
    st.session_state.messages = []


st.title("📚 AI Document Assistant")

st.caption(
    "Upload documents, build a reusable FAISS knowledge base, "
    "and ask grounded questions with Groq."
)

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    st.divider()
    st.subheader("Google Drive")

    drive_link = st.text_input(
        "Paste a shared Drive file or folder link",
        placeholder="https://drive.google.com/...",
    )

    if st.button(
        "Load Drive link",
        use_container_width=True,
    ):
        if not drive_link.strip():
            st.warning(
                "Paste a Google Drive link first."
            )
        else:
            try:
                with st.spinner(
                    "Loading from Google Drive..."
                ):
                    drive_files = load_drive_link(
                        drive_link.strip()
                    )

                supported = [
                    item
                    for item in drive_files
                    if Path(item[1]).suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ]

                if not supported:
                    st.error(
                        "No supported PDF, DOCX, "
                        "TXT or MD file was found."
                    )
                else:
                    st.session_state.drive_files = (
                        supported
                    )
                    st.success(
                        f"Loaded {len(supported)} "
                        "Drive file(s)."
                    )

            except Exception as exc:
                st.error(str(exc))

    if st.session_state.drive_files:
        st.caption("Loaded from Drive:")

        for _, filename in (
            st.session_state.drive_files
        ):
            st.write(f"• {filename}")

        if st.button(
            "Clear Drive files",
            use_container_width=True,
        ):
            st.session_state.drive_files = []
            st.session_state.processed_signature = None
            st.session_state.document_data = None
            st.session_state.messages = []
            st.rerun()

    st.divider()

    if st.button(
        "Clear knowledge base",
        use_container_width=True,
    ):
        st.session_state.processed_signature = None
        st.session_state.document_data = None
        st.session_state.drive_files = []
        st.session_state.messages = []
        st.rerun()


all_files = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        all_files.append((
            uploaded_file.name,
            uploaded_file.getvalue(),
        ))

all_files.extend(
    st.session_state.drive_files
)

if all_files:
    signature = files_signature(all_files)

    if (
        signature
        != st.session_state.processed_signature
    ):
        try:
            with st.spinner(
                "Extracting, chunking and embedding "
                "documents..."
            ):
                st.session_state.document_data = (
                    process_documents(all_files)
                )

                st.session_state.processed_signature = (
                    signature
                )

                st.session_state.messages = []

        except Exception as exc:
            st.session_state.document_data = None
            st.error(
                f"Document processing failed: {exc}"
            )


data = st.session_state.document_data

if data:
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Documents",
        len(data["file_info"]),
    )

    col2.metric(
        "Extracted sections",
        len(data["records"]),
    )

    col3.metric(
        "Chunks",
        len(data["chunks"]),
    )

    with st.expander(
        "📄 Extracted document information",
        expanded=True,
    ):
        for info in data["file_info"]:
            st.markdown(
                f"**{info['filename']}**"
            )

            st.caption(
                f"Type: {info['type']} | "
                f"Extracted sections/pages: "
                f"{info['pages_or_sections']} | "
                f"Characters: "
                f"{info['characters']:,}"
            )

    st.subheader("💬 Ask your documents")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                with st.expander(
                    "Retrieved sources"
                ):
                    for i, source in enumerate(
                        message["sources"],
                        start=1,
                    ):
                        page_text = (
                            f" · Page {source['page']}"
                            if source.get("page")
                            else ""
                        )

                        st.markdown(
                            f"**{i}. "
                            f"{source['filename']}"
                            f"{page_text}** "
                            f"· score "
                            f"{source['score']:.3f}"
                        )

                        st.write(source["text"])
                        st.divider()

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        retrieved = hybrid_search(
            question,
            data["index"],
            data["chunks"],
            top_k=5,
        )

        try:
            with st.chat_message("assistant"):
                with st.spinner(
                    "Searching documents "
                    "and asking Groq..."
                ):
                    answer = ask_groq(
                        question,
                        retrieved,
                    )

                st.markdown(answer)

                with st.expander(
                    "Retrieved sources",
                    expanded=True,
                ):
                    for i, source in enumerate(
                        retrieved,
                        start=1,
                    ):
                        page_text = (
                            f" · Page {source['page']}"
                            if source.get("page")
                            else ""
                        )

                        st.markdown(
                            f"**{i}. "
                            f"{source['filename']}"
                            f"{page_text}** "
                            f"· score "
                            f"{source['score']:.3f}"
                        )

                        st.write(source["text"])
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": retrieved,
            })

        except Exception as exc:
            st.error(str(exc))

else:
    st.info(
        "Upload one or more documents from the sidebar, "
        "or load a shared Google Drive file."
    )

    st.markdown(
        """
        **Pipeline:** Extract → Chunk → Sentence Transformers → FAISS →
        Hybrid retrieval → Groq answer → Sources
        """
    )
