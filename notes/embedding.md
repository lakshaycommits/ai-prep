# RAG Embedding Models — Interview Notes

## What an embedding model actually does

An embedding model maps variable-length text → a fixed-length dense vector of floats.

```
"flight cancellation refund policy" → [0.023, -0.847, 0.412, ..., 0.109]  # 1536 floats
```

The core property: **semantically similar texts produce geometrically close vectors**. "refund for cancelled flight" and "money back on cancelled booking" end up near each other in vector space even though they share zero words.

This is what makes semantic search work — and what fundamentally differentiates it from keyword (BM25) search.

---

## Architecture: how embedding models are trained

Most modern embedding models are transformer encoders fine-tuned with **contrastive learning**.

### The training objective

Given a pair (query, relevant_document), train the model so that:
- `cosine_similarity(embed(query), embed(doc))` → high (close to 1.0)
- `cosine_similarity(embed(query), embed(random_doc))` → low (close to 0.0)

This is called **metric learning** — the model learns a geometric space where meaning = proximity.

### Bi-encoder architecture (what all practical embedding models use)

```
Query  → [Encoder] → query_vector
Document → [Encoder] → doc_vector
similarity = cosine(query_vector, doc_vector)
```

- Query and document are encoded **independently**
- Similarity is computed **after** encoding
- This means you can pre-compute and store all document vectors at index time
- Query time cost = one forward pass for the query + dot product over stored vectors
- **This is why embedding-based search scales** — document encoding is offline

### Cross-encoder architecture (used in re-ranking, not retrieval)

```
[Query + Document concatenated] → [Encoder] → similarity_score
```

- Query and document are encoded **together** — the model sees both simultaneously
- Full attention between query and document tokens → much higher accuracy
- Cannot pre-compute document vectors — must encode query+doc pair at query time
- **Does not scale for retrieval** — if you have 1M docs, you'd need 1M forward passes per query
- Used only for **re-ranking** a small top-K (e.g., re-rank top-50 retrieved chunks)

> Interview insight: "Why not use a cross-encoder for retrieval?" — because you'd need to run N forward passes per query where N = corpus size. A bi-encoder runs 1 forward pass per query and uses pre-computed document vectors.

---

## The three numbers that define an embedding model

### 1. Embedding dimension (d)

The length of the output vector. Common values: 384, 768, 1024, 1536, 3072.

Higher dimension = more expressive geometric space = better at capturing nuance.

But: higher dimension = more storage, higher VRAM, slower similarity search.

```python
# Storage calculation
num_chunks = 1_000_000
dim = 1536                  # text-embedding-3-large
bytes_per_float = 4         # float32
storage_gb = (num_chunks * dim * bytes_per_float) / 1e9
# = 6.14 GB just for vectors, before metadata
```

For 10M chunks at 1536d: ~61 GB of vectors. This is why dimension matters in production.

### 2. Max input tokens (context window)

The maximum number of tokens the model can process in one pass. Text beyond this limit is **silently truncated** — not an error, no warning, just dropped.

| Model | Max tokens |
|---|---|
| all-MiniLM-L6-v2 | 256 |
| BGE-small/base/large | 512 |
| text-embedding-3-small | 8191 |
| text-embedding-3-large | 8191 |
| Cohere Embed v3 | 512 |
| E5-mistral-7b | 4096 |

**Silent truncation is a production bug.** If your chunks are 600 tokens and your model maxes at 512, the last 88 tokens of every chunk are silently dropped. Check this explicitly.

```python
import tiktoken
enc = tiktoken.encoding_for_model("text-embedding-3-small")

def safe_embed(text: str, max_tokens: int = 8191) -> list[float]:
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        # Truncate or raise, never silently pass
        text = enc.decode(tokens[:max_tokens])
    return embed(text)
```

### 3. Training data and domain

A model trained on Wikipedia + CommonCrawl performs well on general text but poorly on medical jargon, legal boilerplate, or aviation terminology. The embedding space it learned doesn't include those concepts at meaningful resolution.

Domain specialization matters when your corpus has:
- Heavy technical jargon
- Non-English language content
- Code (most general models struggle with code semantics)
- Short texts (product names, entity IDs) — trained on sentence pairs, not tokens

---

## Major embedding model families

### OpenAI text-embedding-3 (small / large)

| Property | small | large |
|---|---|---|
| Dimensions | 1536 | 3072 |
| Max tokens | 8191 | 8191 |
| Pricing | $0.02 / 1M tokens | $0.13 / 1M tokens |

- Hosted API — no GPU required
- Supports **Matryoshka** (dimensions reducible without retraining)
- Best general-purpose quality for English text
- `text-embedding-3-large` is the strongest OpenAI option, competitive with fine-tuned open models
- Closed source — data stays with OpenAI (matters for enterprise/private deployments)

```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",
    input="flight cancellation policy",
    dimensions=512   # Matryoshka truncation
)
vector = response.data[0].embedding
```

### Cohere Embed v3

- Strong multilingual support (100+ languages, same model)
- Explicitly designed for retrieval (separate `input_type` for queries vs documents)
- Supports both float and binary/int8 quantized output

```python
import cohere
co = cohere.Client()

doc_embeddings = co.embed(
    texts=chunks,
    model="embed-english-v3.0",
    input_type="search_document"   # critical — use "search_query" for queries
).embeddings

query_embedding = co.embed(
    texts=["what is the refund policy"],
    model="embed-english-v3.0",
    input_type="search_query"
).embeddings[0]
```

> The `input_type` distinction is a Cohere-specific design — the model uses asymmetric embeddings for queries vs documents, which improves retrieval quality. Forgetting this is a common mistake.

### BGE (BAAI General Embedding)

Open-source family from Beijing Academy of AI. Runs locally, zero API cost.

| Model | Dim | MTEB Score | Notes |
|---|---|---|---|
| BGE-small-en-v1.5 | 384 | 62.17 | Fastest, smallest |
| BGE-base-en-v1.5 | 768 | 63.55 | Good balance |
| BGE-large-en-v1.5 | 1024 | 64.23 | Best in family |
| BGE-M3 | 1024 | 66+ | Multilingual, dense+sparse hybrid |

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# BGE requires a query prefix for retrieval tasks
query = "Represent this sentence for searching relevant passages: " + user_query
query_vec = model.encode(query, normalize_embeddings=True)
doc_vecs = model.encode(chunks, normalize_embeddings=True, batch_size=32)
```

> The query prefix is mandatory for BGE retrieval tasks. Without it, scores drop significantly. Document chunks don't need the prefix.

### E5 (Microsoft)

Family focused on instruction-tuned embeddings.

- `e5-large-v2`: strong general retrieval
- `e5-mistral-7b-instruct`: LLM-based embedder, 4096 token window, top MTEB scores but expensive to run (7B params)
- Uses "query: " and "passage: " prefixes (similar to BGE)

### GTE (Alibaba / Tongyi)

- `gte-large`: competitive with BGE-large
- `gte-qwen2-7b-instruct`: recent top performer on MTEB

### Voyage AI

- Strong for code and technical content
- `voyage-code-2`: purpose-built for code retrieval
- `voyage-large-2-instruct`: top general-purpose performance

### When to use open-source vs. API

| Factor | Use API (OpenAI/Cohere) | Use open-source (BGE/E5) |
|---|---|---|
| Data privacy | Data leaves your infra | Fully on-premise |
| GPU availability | No GPU needed | Need inference infra |
| Cost at scale | Can be expensive (per token) | One-time infra cost |
| Domain specialization | Hard to fine-tune | Can fine-tune on your data |
| Latency | Network round-trip | Local inference |

---

## Matryoshka Representation Learning (MRL)

This is the most commonly asked advanced embedding topic in interviews.

### The core idea

Traditional embedding models: change the dimension → retrain from scratch.

Matryoshka models: train once, truncate to any smaller dimension, quality degrades gracefully.

Named after Russian nesting dolls — the full 1536-dim vector "contains" usable 768-dim, 512-dim, 256-dim versions inside it.

### How it's trained

Instead of a single loss on the full embedding, MRL uses a **multi-scale loss**:

```
total_loss = L(full_dim) + L(first_half) + L(first_quarter) + L(first_eighth)
```

The model is forced to pack the most important information into the first few dimensions, and progressively add finer-grained detail in later dimensions.

At inference time:
```python
full_vector = embed("some text")         # [d1, d2, ..., d1536]
half_vector = full_vector[:768]          # valid, usable embedding
quarter_vector = full_vector[:384]       # also valid — quality degraded gracefully
```

This only works because the model was trained with MRL. Truncating a standard non-MRL model gives garbage.

### Why this matters in production

```python
# Scenario: 50M document corpus
# text-embedding-3-large: 3072 dims
# Storage at full dim: 50M * 3072 * 4 bytes = 614 GB

# With Matryoshka truncation to 256 dims:
# Storage: 50M * 256 * 4 bytes = 51 GB
# Quality loss: ~5-8% on MTEB benchmarks
# 12x storage reduction

from openai import OpenAI
client = OpenAI()

# OpenAI natively supports dimension reduction via API
response = client.embeddings.create(
    model="text-embedding-3-large",
    input=chunks,
    dimensions=256   # any value ≤ 3072
)
```

### MRL trade-off curve (text-embedding-3-large)

| Dimensions | Relative MTEB score | Storage vs full |
|---|---|---|
| 3072 (full) | 100% | 1x |
| 1536 | ~98.5% | 0.5x |
| 768 | ~97% | 0.25x |
| 256 | ~93% | 0.08x |
| 64 | ~84% | 0.02x |

**Interview answer:** "I'd use Matryoshka truncation to 512 or 768 dims for most production systems — you get 3–4x storage reduction with under 3% quality loss. Only use full dimensions if MTEB benchmarks on your specific domain justify it."

### Which models support MRL

- OpenAI `text-embedding-3-small` and `text-embedding-3-large` — yes, native API support
- Cohere Embed v3 — no native MRL, but int8/binary quantization available
- BGE family — some variants support MRL; `bge-en-icl` does
- `nomic-embed-text-v1.5` — open-source MRL model, strong for local deployments

---

## Evaluating embedding models: MTEB

**MTEB (Massive Text Embedding Benchmark)** is the standard benchmark for embedding models. Published by Hugging Face. Covers 56 datasets across 8 tasks: retrieval, clustering, classification, summarization, etc.

```
https://huggingface.co/spaces/mteb/leaderboard
```

The **Retrieval** subset (15 datasets, including MSMARCO, BEIR, HotpotQA) is what matters most for RAG. Sort by "Retrieval" score, not overall MTEB.

**Important caveat:** MTEB is general-domain English text. If your corpus is medical, legal, code, or non-English — MTEB ranking is a starting point, not a guarantee. Always eval on your own data.

### How to evaluate on your own data

```python
from sentence_transformers.evaluation import InformationRetrievalEvaluator

# You need: queries, corpus, and relevance labels
evaluator = InformationRetrievalEvaluator(
    queries={"q1": "what is the refund policy"},
    corpus={"d1": "...", "d2": "..."},
    relevant_docs={"q1": {"d1"}},
    score_functions={"cos_sim": cos_sim}
)

# Returns Recall@1, Recall@5, Recall@10, MRR@10, NDCG@10
results = evaluator(model)
```

---

## Asymmetric vs. symmetric retrieval

**Symmetric:** Query and document are similar in length and style. "What is the refund policy?" ↔ "The refund policy allows cancellations within 24 hours."

**Asymmetric:** Query is short (a question), document is long (a paragraph with the answer). Most RAG use cases are asymmetric.

Models trained on symmetric pairs (e.g., simple sentence-pair datasets) perform poorly on asymmetric retrieval. Models like BGE, E5, and Cohere Embed v3 are explicitly trained for asymmetric retrieval with query/passage prefixes or `input_type`.

Always check: was this model trained for asymmetric retrieval?

---

## Quantization: reducing vector memory

Beyond MRL, you can reduce storage by quantizing the float32 values.

| Type | Bits per dim | Storage vs float32 | Quality loss |
|---|---|---|---|
| float32 | 32 | 1x | baseline |
| float16 | 16 | 0.5x | ~0% |
| int8 | 8 | 0.25x | ~1% |
| binary | 1 | 0.03x | ~5–10% |

```python
# Qdrant supports all quantization types natively
from qdrant_client.models import ScalarQuantizationConfig, QuantizationType

client.create_collection(
    "my_collection",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    quantization_config=ScalarQuantizationConfig(
        type=QuantizationType.INT8,
        quantile=0.99,
        always_ram=True
    )
)
```

Combining MRL (256 dims) + int8 quantization gives ~50x storage reduction vs. full float32 at 1536 dims.

---

## Similarity metrics

### Cosine similarity

```
cosine(a, b) = (a · b) / (|a| * |b|)
```

- Range: -1 to 1 (for unnormalized), 0 to 1 (for L2-normalized)
- Scale-invariant — only cares about direction, not magnitude
- **Most common for text embeddings**
- If vectors are L2-normalized (which most models do), cosine similarity = dot product

### Dot product

```
dot(a, b) = a · b = Σ(aᵢ * bᵢ)
```

- Faster to compute than cosine (no normalization step)
- Equivalent to cosine for normalized vectors
- Used in FAISS and most ANN indexes by default

### L2 (Euclidean distance)

```
L2(a, b) = sqrt(Σ(aᵢ - bᵢ)²)
```

- Lower = more similar (opposite convention from cosine)
- Magnitude-sensitive — models that output different-magnitude vectors need this
- Less common for text, more common for image embeddings

> In Qdrant: `Distance.COSINE`, `Distance.DOT`, `Distance.EUCLID` — always match this to what your embedding model was trained with.

---

## The embedding pipeline in production

```
[Document]
    ↓
[Preprocessing: normalize whitespace, strip HTML, language detection]
    ↓
[Chunking: your strategy from the chunking notes]
    ↓
[Token count check: assert len(tokens) ≤ model.max_tokens]
    ↓
[Batch embedding: model.encode(chunks, batch_size=32)]
    ↓
[Optional: L2-normalize vectors if model doesn't do this automatically]
    ↓
[Upsert to Qdrant with metadata payload]
```

### Batching matters

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

# Wrong: one call per chunk
vecs = [model.encode(chunk) for chunk in chunks]   # slow

# Right: batch encode
vecs = model.encode(
    chunks,
    batch_size=64,         # tune to GPU VRAM
    show_progress_bar=True,
    normalize_embeddings=True
)
```

Batch encoding is 10–50x faster than one-at-a-time because it parallelizes the forward pass across the batch.

---

## How to choose an embedding model — decision framework

### Step 1: Define constraints

- Is your data private? → Open-source only (BGE, E5, GTE)
- Do you have GPU infra? → If no, use API (OpenAI, Cohere, Voyage)
- What language? → Multilingual: Cohere Embed v3 or BGE-M3
- What domain? → Code: Voyage Code 2; Medical/Legal: domain-specific fine-tune

### Step 2: Check MTEB Retrieval leaderboard

Filter to models that fit your constraints. Sort by Retrieval score. Note max token limit — it must be ≥ your target chunk size.

### Step 3: Shortlist 2–3 models

Run against a small golden dataset: ~100 questions with known relevant chunks. Measure Recall@5 and MRR@10.

### Step 4: Factor in cost and latency

```
API cost = (total_corpus_tokens / 1M) * price_per_million
```

For 10M chunks × 256 tokens = 2.56B tokens:
- `text-embedding-3-small` at $0.02/1M = $51 one-time indexing
- `text-embedding-3-large` at $0.13/1M = $333 one-time indexing

Re-embedding cost matters when you update your corpus frequently.

### Step 5: Consider fine-tuning

If MTEB leaders still miss your domain (medical, legal, proprietary jargon):

```python
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
from sentence_transformers import InputExample

# Prepare training pairs: (query, positive_doc, negative_doc)
train_examples = [
    InputExample(texts=["flight PNR status", "PNR confirmation details page", "hotel booking confirmation"]),
]

model = SentenceTransformer("BAAI/bge-large-en-v1.5")
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.TripletLoss(model)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
)
model.save("bge-large-indigo-finetuned")
```

For IndiGo-specific jargon (PNR, sector, fare bucket, GDS, availability classes), a fine-tuned BGE would outperform generic OpenAI embeddings.

---

## Common interview questions + strong answers

**Q: "Why does embedding dimension matter?"**
> Dimension determines the resolution of the semantic space. Higher dimension = more expressive, can separate more nuanced concepts. But it's not always worth it — you get diminishing returns past 768d for most tasks, and storage/compute cost scales linearly. With Matryoshka models, you can pick the dimension that gives acceptable quality for your storage budget.

**Q: "What happens when a chunk exceeds the model's max token limit?"**
> It's silently truncated — no error is raised. The embedding only represents the first N tokens. This is a silent production bug that destroys retrieval for long chunks. Always validate token counts before embedding and either truncate explicitly or split further.

**Q: "When would you fine-tune an embedding model?"**
> When your domain has vocabulary the base model hasn't seen at useful resolution — medical terminology, legal boilerplate, proprietary product names, non-English technical text. Also when your retrieval pattern is highly asymmetric and the base model's training distribution doesn't match. Fine-tuning requires (query, positive_doc, negative_doc) triplets — generating these from user click logs or existing labeled data is the key challenge.

**Q: "Cosine similarity vs dot product — when does it matter?"**
> For L2-normalized vectors (which most modern models produce), they're mathematically equivalent. Dot product is faster to compute. The difference only matters if your model doesn't normalize outputs — in that case, a high-magnitude irrelevant vector could score higher than a correctly similar low-magnitude vector with dot product, while cosine would correctly rank the similar one higher.

**Q: "What's Matryoshka representation learning and when would you use it?"**
> MRL trains a model so the first K dimensions of any embedding are themselves a valid embedding — quality degrades gracefully as you reduce dimensions. Useful when storage is the bottleneck: a 1536-dim OpenAI vector corpus takes 6x more space than a 256-dim version with only ~7% quality loss. I'd use it for any corpus over ~10M chunks or any latency-sensitive system where smaller vectors mean faster ANN search.

---

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| Good model, poor retrieval on technical queries | Domain mismatch | Fine-tune on domain data or use domain-specific model |
| Last part of chunk never retrieved | Silent truncation at max tokens | Validate token count before embedding |
| Query and document vectors in different spaces | Wrong model / using same model for both | Use same model for query and document; use input_type for Cohere |
| Multilingual queries fail | English-only model | Switch to Cohere Embed v3 or BGE-M3 |
| Storage too large at scale | Full-dim float32 | Apply MRL truncation + int8 quantization |
| Near-duplicate results in top-K | Overlapping chunks embedded too similarly | Reduce overlap, or add MMR (Maximal Marginal Relevance) diversification |

---

## Key terms for interviews

- **Bi-encoder** — encodes query and document independently; scales for retrieval
- **Cross-encoder** — encodes query+document together; better accuracy, used only for re-ranking
- **MTEB** — the standard benchmark for embedding models; sort by Retrieval score for RAG
- **Matryoshka Representation Learning (MRL)** — training technique enabling dimension truncation without quality collapse
- **Silent truncation** — text beyond max token limit is dropped silently; a production bug
- **Asymmetric retrieval** — short query ↔ long document; most RAG use cases; requires models trained for this
- **Input type** — Cohere's mechanism for asymmetric embeddings; "search_query" vs "search_document"
- **int8 quantization** — 4x storage reduction, ~1% quality loss
- **Contrastive learning** — the training objective that creates semantic vector spaces
- **L2 normalization** — scaling vectors to unit length; makes cosine = dot product
