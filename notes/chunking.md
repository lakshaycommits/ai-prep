# RAG Chunking Strategies — Interview Notes

## What is chunking and why it matters

Embedding models have a fixed input window (typically 512–8192 tokens). Documents are almost always longer. Chunking breaks documents into embeddable pieces.

**Core tension:** Granularity vs. Context
- Smaller chunks → more precise retrieval, less context for LLM
- Larger chunks → more context for LLM, less precise retrieval

Wrong chunking silently destroys retrieval quality — the embedding model and vector store can be perfect, but if chunks straddle key answers, the system fails.

---

## Strategy 1: Fixed-size chunking

Split text at exactly N tokens/characters regardless of meaning.

```python
def fixed_chunk(text: str, size: int = 512) -> list[str]:
    return [text[i:i+size] for i in range(0, len(text), size)]
```

**How it works:** Tokenize the document, slice every N tokens, done.

**Pros:**
- Simplest to implement — O(1) compute
- Deterministic, reproducible
- No NLP dependencies

**Cons:**
- Cuts mid-sentence, mid-concept
- Context loss at every boundary
- Zero semantic awareness — "the system queries" ends one chunk, "an availability cache" starts the next — neither is retrievable alone

**Use when:**
- Log files, records with fixed schema
- Quick prototype / baseline
- Data where meaning is self-contained per line

**Chunk size heuristics:**
- 256–512 tokens: precision-focused QA
- 512–1024 tokens: summarization tasks
- Always stay under embedding model's max input window

---

## Strategy 2: Recursive character splitting

Try to split on `\n\n` (paragraphs) first, then `.` / `!` / `?` (sentences), then ` ` (words), then characters. Only go finer if the chunk is still too large.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)
```

**How it works:** Hierarchical fallback — respects natural text boundaries wherever possible.

**Pros:**
- Respects sentence and paragraph boundaries
- Simple to implement, widely supported
- No embedding model required at index time

**Cons:**
- Assumes paragraph = topic unit (fragile for dense docs)
- Still not semantically aware — adjacent sentences on different topics may land in the same chunk
- Chunk size still bounded by character count, not meaning

**Use when:**
- General prose (blogs, articles, internal wikis)
- Product documentation
- Default choice when document structure is unknown
- Most real-world production RAG systems use this as the baseline

---

## Strategy 3: Semantic chunking

Embed every sentence. Compute cosine similarity between adjacent sentences. When similarity drops below a threshold → topic shift → chunk boundary.

```python
from llama_index.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings import OpenAIEmbedding

splitter = SemanticSplitterNodeParser(
    buffer_size=1,
    breakpoint_percentile_threshold=95,
    embed_model=OpenAIEmbedding()
)
nodes = splitter.get_nodes_from_documents(documents)
```

**How it works:**
1. Sentence-tokenize the document
2. Embed each sentence
3. Compute cosine similarity between sentence[i] and sentence[i+1]
4. Where similarity < threshold → insert chunk boundary
5. Group sentences between boundaries into one chunk

**Pros:**
- Chunks represent coherent topics, not arbitrary character windows
- Best retrieval precision of all strategies
- Handles long and short sentences equally well — boundary is semantic, not length-based

**Cons:**
- Requires an embedding call per sentence at index time → 10–50x slower than fixed-size
- Non-deterministic chunk boundaries (threshold-sensitive)
- If threshold is wrong, you get either too many tiny chunks or too few large ones
- Adds embedding model as a build-time dependency

**Use when:**
- Research papers, textbooks, dense knowledge bases
- High-quality QA systems where retrieval precision matters most
- Domains with rapid topic shifts (medical literature, legal documents)

**Key parameter — breakpoint threshold:**
- Low threshold (e.g., 80th percentile) → more chunk boundaries → smaller, more precise chunks
- High threshold (e.g., 95th percentile) → fewer boundaries → larger chunks with more context
- Tune this against Recall@K on a held-out eval set

---

## Strategy 4: Document-aware chunking

Split on the document's own structure: Markdown headings, HTML tags, PDF section headings, code function/class boundaries, JSON keys.

```python
# Markdown: split on headings
import re

def markdown_chunks(text: str) -> list[dict]:
    sections = re.split(r'\n(?=#{1,3} )', text)
    return [{"content": s, "heading": s.split('\n')[0].strip('#').strip()} for s in sections]

# Code: use AST (never regex for code)
import ast

def python_function_chunks(source: str) -> list[dict]:
    tree = ast.parse(source)
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            chunk_text = ast.get_source_segment(source, node)
            chunks.append({"content": chunk_text, "name": node.name, "type": type(node).__name__})
    return chunks
```

**How it works:** Use a format-specific parser to extract structural units. One heading section = one chunk. One function = one chunk. One table = one chunk.

**Pros:**
- Chunks are complete units of meaning, not arbitrary splits
- Metadata extraction is natural (section title, function name, file path)
- Highest quality for structured documents

**Cons:**
- Requires format-specific parser for each document type
- Breaks on inconsistently structured documents
- Not suitable for unstructured prose with no markup

**Use when:**
- Codebases (split on function/class boundaries via AST)
- Markdown/RST documentation
- PDFs with clear section headers
- API reference docs, OpenAPI specs
- FAQ pages (each Q&A pair = one chunk)

**Metadata to always attach:**
```python
{
    "source_file": "pricing_engine.py",
    "section": "DynamicPricer",
    "chunk_type": "function",
    "chunk_name": "calculate_fare",
    "page_number": 12   # for PDFs
}
```

---

## Strategy 5: Sliding window with overlap

Like fixed-size, but adjacent chunks share N tokens. Chunk 1 covers tokens 0–100, chunk 2 covers tokens 75–175, overlap is 25 tokens.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100   # ~20% overlap
)
chunks = splitter.split_text(document)
```

**How it works:** Slide a window across the document, advancing by (chunk_size - overlap) tokens each step.

**Pros:**
- Prevents answer-straddling-boundary failures
- Easy to layer on top of any other strategy
- Critical answers no longer fall through the cracks

**Cons:**
- 20–40% more vectors stored
- Duplicate content in multiple chunks → retrieval may return near-identical chunks
- LLM sees same content twice → wasted context window

**Use when:**
- Legal or compliance documents (cannot miss a clause)
- Dense technical text where key facts span sentences
- Any strategy as a safety net — always add some overlap

**Overlap heuristic:** 10–20% of chunk size. For 512-token chunks, use 50–100 token overlap.

---

## Advanced pattern: Parent-child chunking

Solves the granularity vs. context tension directly.

**Mechanism:**
- Store **small chunks** (128 tokens) as vectors → high retrieval precision
- When returning to LLM, fetch the **parent chunk** (512 tokens) instead → full context

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.vectorstores import Qdrant

child_splitter = RecursiveCharacterTextSplitter(chunk_size=128)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

retriever = ParentDocumentRetriever(
    vectorstore=qdrant_store,
    docstore=InMemoryStore(),   # or Redis for production
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

**With Qdrant specifically:**
- Store parent text in the `payload` field of child vectors
- On retrieval, look up `payload["parent_id"]` and return parent text
- No separate parent vector needed — parents live in child payloads

**Why this matters in interviews:** This is the pattern that most clearly shows you understand the retrieval-generation tension. Mention it proactively.

---

## Decision framework — the interview answer

| Document type | Recommended strategy | Reasoning |
|---|---|---|
| 100-page PDF (structured sections) | Document-aware | Split on headings; section = unit of meaning |
| 100-page PDF (continuous prose) | Semantic + overlap | Topic-coherent chunks; overlap prevents boundary loss |
| Codebase | Document-aware (AST) | Function/class is the natural retrieval unit |
| FAQ document | Document-aware | Each Q&A pair is one chunk, no exceptions |
| Internal wiki / Notion | Recursive + overlap | Markdown structure, but inconsistent — recursive is safe default |
| Legal/compliance doc | Sliding window (large overlap) | Cannot miss clauses that straddle boundaries |
| Emails / Slack threads | Fixed-size or per-message | Each message is already a coherent unit |

---

## What interviewers are really testing

1. **Do you know that chunking affects retrieval, not just storage?** Most candidates think chunking is just "how to fit text into an embedding model." It directly determines what your retriever returns.

2. **Can you diagnose a retrieval failure?** If recall is low, the first thing to check is chunk boundaries — not the embedding model.

3. **Do you know the parent-child pattern?** This separates senior candidates from junior. Mention it unprompted.

4. **Can you reason about trade-offs?** There is no universally correct chunk size. It's a function of: embedding model window, LLM context window, query type, and document structure.

---

## Failure modes to know

| Symptom | Likely chunking cause | Fix |
|---|---|---|
| Answer is in the docs but never retrieved | Key info split across two chunks | Add overlap, or use semantic/doc-aware chunking |
| Retrieved chunks are irrelevant | Chunks too large, pulling in noise | Reduce chunk size, use semantic splitting |
| Correct chunk retrieved but LLM gives wrong answer | Chunk too small, missing surrounding context | Use parent-child pattern or increase chunk size |
| Duplicate results in top-K | Overlap too high | Reduce overlap, add deduplication step |
| Great in dev, poor in prod | Document structure changed | Make chunking strategy robust to structural variance |

---

## Key terms to use in interviews

- **Chunk boundary problem** — when the answer straddles two chunks
- **Semantic coherence** — chunks should represent a single topic
- **Retrieval granularity** — smaller chunks = more precise retrieval
- **Parent-child retrieval** — small chunks for retrieval, large chunks for generation
- **Overlap** — shared tokens between adjacent chunks to avoid boundary loss
- **AST-based splitting** — parsing code structure instead of splitting on characters
- **Breakpoint threshold** — cosine similarity drop that triggers a new chunk in semantic splitting
