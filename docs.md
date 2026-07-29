# Maujmasti NumPy LLM 

This project implements a complete generative pre-trained transformer (GPT) style language model from scratch, using only NumPy.

## Architecture

The model follows a standard decoder-only transformer architecture with the following components:

- **BBPETokenizer:** A byte-pair encoding tokenizer written from scratch to process raw UTF-8 text into integer tokens.
- **Embedding & PositionalEncoding:** Maps discrete token IDs into continuous vectors and applies sinusoidal positional encodings. 
- **Transformer Encoder Blocks**:
  - `MultiHeadAttention`: Computes self-attention over the sequence with a causal mask to ensure autoregressive properties.
  - `LayerNormalization`: Normalizes inputs for stable training.
  - `FeedForwardNetwork`: MLP block for representation expansion (typically 4x embedding dimension).
- **CrossEntropyLoss:** Numerically stable cross-entropy loss calculation tailored for language modeling.

Everything, including the Forward Pass and the complex Multi-Head Attention Backward Pass, has been meticulously implemented using only `np.dot`, `np.sum`, and other primitive NumPy operations—no PyTorch or TensorFlow!

## Getting Started

### Prerequisites

All you need is Python and NumPy. We recommend using a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy
```

### Training

To begin training the model on the provided `tiny-shakespeare.txt` dataset, simply execute:

```bash
python train.py
```

This will:
1. Initialize the BBPETokenizer and train it on the raw text data.
2. Initialize the PyTorch-equivalent dataloaders to batch the tokenized inputs.
3. Build the `GPTDecoder` model.
4. Execute the training loop using our custom `AdamOptimizer` and `WarmupCosineScheduler`.
5. Upon finishing, the model will generate a sample token sequence.

## CPU Training Time Estimates (Intel i5-1240p)

Training deep neural networks strictly utilizing CPU array operations is computation-intensive. The Intel Core i5-1240p is a capable CPU (12 cores, 16 threads), but training times will still scale tightly with your hyperparameters.

For the default **Small Configuration** present in `train.py`:
- `vocab_size` = 500
- `embed_dim` = 64
- `num_layers` = 2, `num_heads` = 4
- `batch_size` = 8, `max_seq_len` = 32
**Estimated Time:** ~3 to 4 minutes per epoch.

For a **Medium Configuration** (more realistic for text generation):
- `vocab_size` = 2000
- `embed_dim` = 128
- `num_layers` = 4, `num_heads` = 4
- `batch_size` = 32, `max_seq_len` = 128
**Estimated Time:** ~2-3 hours per epoch. The matrix multiplications (particularly the QK^T attention scores and MLP projections) scale cubically, which will saturate CPU vectors quickly. 

> [!TIP]
> For serious model training, you should implement the equivalent functions using `cupy` as originally planned. `cupy` retains the NumPy API but seamlessly executes matrix multiplications on the GPU, accelerating this exact architecture by massive margins (often 50x-100x).
