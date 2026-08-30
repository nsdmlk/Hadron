# LLM-Core

<p align="center">
  <b>Transformer from scratch · Attention · BPE Tokenizer · Training pipeline</b><br>
  <sub>Research project: how LLMs work inside</sub>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-complete-green" alt="status">
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="python">
  <img src="https://img.shields.io/badge/framework-pytorch-orange" alt="pytorch">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="license">
</p>

---

## What is LLM-Core?

LLM-Core is a **research project** that implements the internals of Large Language Models from scratch. It is not a product — it is a learning artifact that answers the question: **"What happens inside an LLM when it reads and writes text?"**

Every component — attention, positional encoding, BPE tokenizer, training loop — is written from zero on PyTorch, with no high-level abstractions.

---

## What was built

### Phase 1: Architecture (completed)

- [X] **Multi-Head Attention** — scaled dot-product attention with Q/K/V projections
- [X] **Positional Encoding** — sinusoidal position embeddings
- [X] **Encoder Layer** — self-attention + feed-forward + residuals + layer norm
- [X] **Decoder Layer** — masked self-attention + cross-attention + feed-forward
- [X] **Transformer Encoder** — stack of N encoder layers
- [X] **Transformer Decoder** — stack of N decoder layers
- [X] **Full Transformer** — encoder-decoder architecture (as in "Attention is All You Need")
- [X] **BPE Tokenizer** — Byte Pair Encoding: train, encode, decode

### Phase 2: Training (completed in Colab)

- [X] **Training pipeline** — batches, gradient clipping, learning rate scheduling
- [X] **Data** — 104M words: Git Pro, Python docs, Strang's books, WikiText-103, ML papers
- [X] **Mini-GPT** — 3.6M parameter model trained on technical text
- [X] **Text generation** — with temperature, top-k sampling, repetition penalty

### Phase 3: Experiments (completed)

- [X] **Overfitting analysis** — why small models memorize instead of generalize
- [X] **Vocabulary filtering** — removing rare words (103k → 26k vocab)
- [X] **Loss curves** — from 7.4 to 1.35 with early stopping
- [X] **Chunked training strategy** — how to train large models on limited compute

---

## What was learned

- How text becomes numbers (tokenization, embeddings)
- How attention links words in context
- Why positional encoding is necessary
- How residual connections stabilize training
- Why large models need billions of words
- What overfitting looks like in language models
- How to design training pipelines

---

## Key results

| Experiment | Result |
|------------|--------|
| Transformer from scratch | ✅ Working encoder-decoder |
| BPE Tokenizer | ✅ Trained on technical text |
| Mini-GPT training | ✅ Loss 1.35 on 2.4M words |
| Generation quality | Technical terms, limited coherence |
| Data collected | 104M words |

---

## Structure

```
LLM-Core/
├── core/
│   ├── attention.py           # Multi-Head Attention
│   ├── embedding.py           # Token embeddings
│   ├── positional_encoding.py # Sinusoidal positions
│   ├── encoder_layer.py       # Transformer encoder layer
│   ├── decoder_layer.py       # Transformer decoder layer
│   ├── transformer_encoder.py # Full encoder
│   ├── transformer_decoder.py # Full decoder
│   ├── transformer.py         # Complete transformer
│   ├── gpt.py                 # Mini-GPT model
│   ├── bpe_tokenizer.py       # BPE implementation
│   ├── mask.py                # Attention masks
│   └── core_math.py           # Softmax, attention scores, layer norm
├── tests/
│   └── test_prompts.py
├── README.md
└── LICENSE
```

---

## How to use

```python
import torch
from core.gpt import GPT
from core.bpe_tokenizer import BPETokenizer

# Train tokenizer
tokenizer = BPETokenizer()
tokenizer.train(texts, num_merges=100)

# Create model
model = GPT(
    vocab_size=len(tokenizer.vocab),
    d_model=128,
    n_heads=4,
    d_ff=512,
    n_layers=2
)

# Generate text
output = model.generate(tokenizer, prompt="git clone", max_len=20)
```

---

## Limitations

This is a **learning project**, not a production LLM:

- Model too small (3.6M vs 175B in GPT-3)
- Limited data (104M vs 45TB in GPT-3)
- No fine-tuning, no RLHF
- No inference optimization

For production use, consider Llama, Mistral, or other open-source LLMs.

---

## Author

**Emelyanov Ilya**
GitHub: [@nsdmlk](https://github.com/nsdmlk)
Telegram: [@KantervilleGhost](https://t.me/KantervilleGhost)

---

## License

MIT © Emelyanov Ilya, 2026

---

<p align="center">
  <sub>Built to understand, not to compete.</sub>
</p>
