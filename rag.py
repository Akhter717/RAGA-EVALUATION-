"""RAG system: PDF -> chunks -> embeddings -> vector store (FAISS) -> retrieval -> Groq answer."""
import io
import os
import re
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from groq import Groq
from pypdf import PdfReader

try:
    import faiss  # vector store
except Exception:  # fall back to a NumPy vector store if FAISS is unavailable
    faiss = None

CHUNK_SIZE = 800     # characters
CHUNK_OVERLAP = 150
DEFAULT_EMBED = "BAAI/bge-small-en-v1.5"

SYSTEM_PROMPT = (
    "You are a concise, accurate assistant. Answer ONLY from the provided context. "
    "After each statement, cite the number of the chunk you used in square brackets, like [1] or [2][3]. "
    "If the answer is not in the context, say \"I don't know.\" and do not cite anything."
)


def rag_model() -> str:
    return os.getenv("RAG_MODEL", "openai/gpt-oss-20b")


def judge_model() -> str:
    return os.getenv("JUDGE_MODEL", "openai/gpt-oss-120b")


def embed_model_name() -> str:
    return os.getenv("EMBED_MODEL", DEFAULT_EMBED)


@dataclass
class Chunk:
    text: str
    source: str
    page: int


# ---------------------------------------------------------------- embeddings
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from fastembed import TextEmbedding
        _embedder = TextEmbedding(model_name=embed_model_name())
    return _embedder


def embed_texts(texts: List[str]) -> np.ndarray:
    vecs = np.array(list(get_embedder().embed(texts)), dtype="float32")
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


# ------------------------------------------------------------------ chunking
def chunk_pdf(file_bytes: bytes, filename: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    reader = PdfReader(io.BytesIO(file_bytes))
    for page_no, page in enumerate(reader.pages, start=1):
        text = " ".join((page.extract_text() or "").split())
        start = 0
        while start < len(text):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) > 50:
                chunks.append(Chunk(piece, filename, page_no))
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


# -------------------------------------------------------------- vector store
class RagIndex:
    def __init__(self, chunks: List[Chunk]):
        if not chunks:
            raise ValueError("No text could be extracted from the PDF(s). Scanned PDFs need OCR.")
        self.chunks = chunks
        self.vecs = embed_texts([c.text for c in chunks])
        self.dim = int(self.vecs.shape[1])
        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.vecs)
            self.store_name = "FAISS (IndexFlatIP, cosine similarity)"
        else:
            self.index = None
            self.store_name = "NumPy in-memory (cosine similarity)"

    def retrieve(self, question: str, k: int) -> List[Tuple[Chunk, float]]:
        k = max(1, min(k, len(self.chunks)))
        q = embed_texts([question])
        if self.index is not None:
            d, i = self.index.search(q, k)
            return [(self.chunks[j], float(s)) for s, j in zip(d[0], i[0]) if j != -1]
        sims = self.vecs @ q[0]
        top = np.argsort(-sims)[:k]
        return [(self.chunks[j], float(sims[j])) for j in top]


# ----------------------------------------------------------------- LLM calls
def chat(model: str, system: str, user: str, temperature: float = 0.0) -> str:
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return (resp.choices[0].message.content or "").strip()


def format_context(chunks: List[Chunk]) -> str:
    """Chunks are numbered [1], [2], ... so the model can cite them."""
    return "\n\n".join(f"[{i}] ({c.source} p.{c.page})\n{c.text}" for i, c in enumerate(chunks, 1))


_CITE = re.compile(r"\s*\[\s*\d+(?:\s*[,;]\s*\d+)*\s*\]")


def strip_citations(text: str) -> str:
    """Remove [1] style markers, so RAGAS scores the plain answer."""
    return _CITE.sub("", text or "").strip()


def answer(question: str, chunks: List[Chunk]) -> str:
    """Answer ONLY from the retrieved chunks."""
    return chat(rag_model(), SYSTEM_PROMPT,
                f"Context:\n{format_context(chunks)}\n\nQuestion: {question}")
