# LLM Internals — Revision Notes
> Scoped for AI/Application Engineers. Agents, RAG, and tool orchestration covered separately.

---

## 1. Transformer Architecture

Transformers transform input (a) to output (b). The architecture has two main blocks: **Encoder** and **Decoder**.

### Flow

```
Raw Text → Tokenizer → Token IDs → Embeddings → Positional Encoding
         → Multi-Head Self-Attention → Add & Norm
         → Feed Forward → Add & Norm
         → [Encoder Output: K, V]

Decoder:
         → Masked Self-Attention
         → Cross-Attention (Q from decoder, K/V from encoder)
         → Feed Forward → Add & Norm
         → Linear + Softmax → next token
```

> **Note:** Tokenizer is a preprocessing step, not part of the encoder itself.

### Add & Norm (Residual Connections)
After every sub-layer (attention, FFN):
```
output = LayerNorm(x + sublayer(x))
```
Prevents vanishing gradients, allows deep stacking.

---

## 2. Tokenization & Vocabulary

- Tokens are numeric representations of text — machines can't understand raw text.
- Every model has its own tokenizer and **vocabulary** (e.g. GPT-4 uses `cl100k` ~100k vocab).

### Vocabulary Size Tradeoffs

| | Large Vocabulary | Small Vocabulary |
|---|---|---|
| Tokens per word | Fewer | More |
| Context window efficiency | Better (more text fits) | Worse |
| Inference cost (you pay) | Lower | Higher |
| Training/hosting cost (provider pays) | Higher (larger embedding matrix) | Lower |

**Two separate costs:**
- **Provider cost** — large vocab = large embedding matrix = expensive to train/host
- **Your cost** — small vocab = more tokens = fills context window faster = expensive at inference

---

## 3. Positional Encoding

Gives each token its position in the sequence (absolute and relative).

- Original transformers: sine/cosine absolute encoding
- Modern models (LLaMA etc.): **RoPE** (Rotary Positional Embedding) — better relative position handling, generalizes to longer sequences

---

## 4. Attention Mechanisms

### Self-Attention vs Cross-Attention

| | Attends to | Q from | K/V from |
|---|---|---|---|
| Self-Attention | Same sequence | Same sequence | Same sequence |
| Cross-Attention | Another sequence | Decoder | Encoder |

**Self-attention** — every token looks at every other token in the same sequence to build context.

**Cross-attention** — decoder attends to encoder output. Used in seq2seq (translation, summarization), Stable Diffusion (text conditioning image generation), RAG.

### Multi-Head Attention
- Run attention `h` times in parallel with different learned projections
- Each head captures different relationships (syntactic, semantic etc.)
- Concatenate outputs → linear projection
- Multi-head is orthogonal to self/cross — both use multi-head in practice

---

## 5. Training vs Inference

### Training (Supervised)
- Decoder receives: `BOS + ground truth tokens` (**teacher forcing**)
- Model predicts next token at each step
- Prediction compared to label ending with `EOS`
- Loss calculated → backprop → weights updated → repeat until loss minimized

### Inference
- Decoder receives: `BOS` only
- Generates tokens autoregressively using its own previous outputs
- Stops at `EOS`

**Teacher forcing** — feeding correct previous tokens during training (not model's own predictions) stabilizes training.

---

## 6. Sampling Strategies

### Temperature
Applied **before** softmax. Controls how peaked or flat the distribution is.

```
low temp  → peaked distribution  → deterministic, factual
high temp → flat distribution    → creative, higher hallucination risk
```

### Softmax
Converts raw logits to probabilities summing to 1. Always happens. Not the same as temperature.

```
Logits → Temperature scaling → Softmax → Top-p / Top-k → Sample
```

### Top-p (Nucleus Sampling)
Applied **after** softmax. Sort tokens by probability, keep smallest set that sums to `p`, sample from that pool.

```
top_p = 0.9 → use tokens whose cumulative probability = 90%
```

Adaptive — small pool when model is confident, large pool when uncertain.

### Top-k
Applied **after** softmax. Always keep exactly `k` highest probability tokens.

```
top_k = 50 → always consider exactly 50 tokens
```

Less adaptive than top-p. Often used together with top-p.

| | Top-p | Top-k |
|---|---|---|
| Cutoff type | Dynamic (cumulative prob) | Fixed (count) |
| Adaptivity | High | Low |
| Preferred in production | Yes | Sometimes combined |

### For Hallucination Reduction
Lower temperature + lower top-p = tighter, more factual outputs.

---

## 7. Hallucination Reduction

Not reducible to 0%, but significantly reducible in practice (~5% target with best practices).

| Strategy | How it helps |
|---|---|
| Low temperature + top-p | Model picks high-probability, confident tokens |
| RAG | Grounds output in retrieved source, reduces reliance on parametric memory |
| Tool use / MCP | Model fetches answers instead of guessing |
| Prompt constraints | Instruct model to say "I don't know" rather than guess |
| Output validation / self-consistency | Run same query multiple times, cross-check outputs |

**Vector RAG vs Vectorless RAG** — hybrid (dense + BM25 keyword) often best for precise lookups like product codes or exact terms.

---

## 8. KV Cache

During inference, every new token must attend to all previous tokens — recomputing K, V matrices each time is wasteful.

KV Cache stores Key and Value matrices from previous steps:

```
Without KV cache:  token 100 recomputes K,V for tokens 1–99 → O(n²)
With KV cache:     token 100 reads stored K,V, computes only for itself → O(n)
```

**Tradeoff:** KV cache lives in GPU memory. Longer context = more memory consumed = higher cost.

---

## 9. Context Window Internals

Attention is **O(n²)** in sequence length:

```
512 tokens   → 512²  = ~262k operations
8k tokens    → 8000² = ~64M operations
128k tokens  → 128k² = ~16B operations
```

Doubling context = 4x compute. Long context = higher cost + higher latency.

### What the industry does about it:
- **Sparse attention** — attend only to nearby + selected tokens, not all
- **Flash Attention** — memory-efficient attention I/O, doesn't reduce complexity but reduces memory bandwidth usage significantly. Used in most production models.
- **RoPE scaling** — lets models generalize beyond their trained context length

---

## 10. Fine-tuning vs Prompting

| | Prompting / RAG | Fine-tuning |
|---|---|---|
| Mechanism | Guide via input | Update model weights |
| Cost | Low (API calls) | High (GPU training) |
| Use case | Fresh/proprietary knowledge | Behavior, tone, format, domain jargon |
| Knowledge injection | Yes (via RAG) | Unreliable — common misuse |
| Reversible | Yes | No (separate model) |

**LoRA (Low Rank Adaptation)** — practical fine-tuning method. Trains two small matrices approximating the weight update instead of updating all weights. Cheaper, same quality. Used under the hood by OpenAI and Azure fine-tuning APIs.

**Rule of thumb:**
- Need fresh or org-specific knowledge → **RAG**
- Need different behavior / output style → **Fine-tune**

---

## 11. Cost & Throughput

### Cost per Token
Published by every provider. Split into:
- **Input tokens** — prompt + context
- **Output tokens** — generated response (more expensive, autoregressively generated)

### Throughput Metrics

| Metric | What it measures |
|---|---|
| TPS (tokens/sec) | Overall speed |
| TTFT (time to first token) | Latency before streaming starts — critical for UX |
| TPOT (time per output token) | Generation speed after first token |

```python
start = time.time()
response = client.messages.create(...)
end = time.time()

tps = (input_tokens + output_tokens) / (end - start)
```

**Factors affecting throughput:**
- Batch size (more concurrent requests → lower TPS per request)
- Input length (long RAG prompts slow prefill)
- Model size (GPT-4o vs GPT-4o-mini is significant)

---

## Quick Reference

```
Tokenizer → Embeddings → Positional Encoding → Attention → FFN → Output

Attention:   Self (same seq) | Cross (encoder→decoder) | Multi-head (parallel)
Sampling:    Temperature → Softmax → Top-p / Top-k → Sample
Training:    Teacher forcing + loss minimization
Inference:   BOS → autoregressive → EOS
KV Cache:    Store K,V to avoid O(n²) recomputation
Context:     O(n²) attention → long context is expensive by math
Fine-tune:   Behavior change (LoRA) | RAG: knowledge grounding
```