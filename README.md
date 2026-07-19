# GPTmodel

A character-level GPT language model built from scratch in PyTorch, trained on Tiny Shakespeare. Implements the full decoder-only Transformer stack — multi-head self-attention, causal masking, feed-forward blocks, residual connections, and layer norm — without relying on pre-built Transformer libraries.

The notebook builds up in two stages: a bigram baseline first, then the full Transformer.

## Architecture

- Token + learned positional embeddings
- Causal (masked) scaled dot-product self-attention, multi-head
- Position-wise feed-forward network (4× expansion, ReLU)
- Pre-norm residual Transformer blocks, stacked `n_layer` times
- Linear language-modeling head over the vocabulary

## Hyperparameters

| Param | Value |
|---|---|
| `batch_size` | 64 |
| `block_size` | 256 |
| `n_embd` | 384 |
| `n_head` | 6 |
| `n_layer` | 6 |
| `dropout` | 0.2 |
| `learning_rate` | 3e-4 |
| `max_iters` | 5000 |

**Parameters**: ~10.79M

## Dataset

[Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) — 1,115,394 characters, 65-character vocabulary, character-level tokenization, 90/10 train/val split.

## Results

| Step | Train Loss | Val Loss |
|---|---|---|
| 0 | 4.28 | 4.28 |
| 1000 | 1.60 | 1.77 |
| 2500 | 1.28 | 1.53 |
| 4000 | 1.15 | 1.48 |
| 4999 | 1.08 | 1.49 |

Validation loss levels off around step 4500 and ticks up slightly by the end — a mild sign of overfitting at this scale.

## Setup

```bash
git clone https://github.com/NansKong/GPTmodel.git
cd GPTmodel
pip install torch numpy pandas matplotlib
jupyter notebook GPT.ipynb
```

The dataset downloads automatically on first run. GPU recommended; falls back to CPU.

## Generate text

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
```

## Limitations

- Character-level tokenization, not subword/BPE
- Small scale (~11M params, ~1M characters) limits coherence over long spans
- No LR scheduling or early stopping
- Single-domain corpus (Early Modern English)

## References

- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., GPT / GPT-2 (OpenAI)
- Karpathy, [char-rnn](https://github.com/karpathy/char-rnn) / Tiny Shakespeare

## License

No license file included yet — add one (e.g. MIT) if you want to permit reuse.
