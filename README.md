# Chhota-NP-LLM

A complete generative pre-trained transformer (GPT / Decoder-Only Transformer) implemented **from scratch** using only primitive vector and matrix array operations with **NumPy** (CPU) and optionally **CuPy** (GPU).

No PyTorch, no TensorFlow, no automatic differentiation framework! Everything from the tokenization, positional encoding, multi-head attention, layer normalization, GELU activation, forward/backward backpropagation passes, Adam optimizer, cross-entropy loss, and warmup-cosine learning rate scheduling is written manually.

---

## Key Features

- **Byte-level BPE Tokenizer (`BBPETokenizer`)**: Built-in Byte Pair Encoding tokenizer trained directly on UTF-8 bytes.
- **Sinusoidal Positional Encoding**: Deterministic positional embeddings capturing relative token positions.
- **Multi-Head Attention & Causal Masking**: Full manual implementation of scaled dot-product multi-head attention with lower-triangular causal masking and backward pass gradients.
- **Pre-LN Transformer Block**: Standard modern GPT architecture with Layer Normalization and Feed-Forward Networks (GELU).
- **Dual Hardware Support (`arr.py`)**: Automatic fallback and seamless execution on CPU via `NumPy` or GPU hardware acceleration via `CuPy`.
- **Adam Optimizer & Cosine Warmup**: Custom Adam optimizer with gradient clipping and Warmup-Cosine learning rate schedule.

---

## ⚙️ Requirements & Setup

This project supports running with [`uv`](https://github.com/astral-sh/uv) (fast Python package installer & resolver) or standard Python `venv` + `pip`.

### Option 1: Using `uv` (Recommended)

#### 1. Quick Run with `uv run` (No installation step needed)
`uv` will automatically set up an environment and run the script:

```bash
# Run training using NumPy / CuPy
uv run train.py

# Run text generation / inference from trained weights
uv run infer.py
```

#### 2. Environment Setup with `uv`
If you prefer creating a virtual environment explicitly:

```bash
# Create a virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate

# Sync dependencies defined in pyproject.toml
uv sync
```

---

### Option 2: Without `uv` (Standard Python `venv` + `pip`)

#### 1. Create and activate a Virtual Environment
```bash
# Create virtual environment (Python >= 3.12 recommended)
python3 -m venv .venv

# Activate the virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (cmd):
# .venv\Scripts\activate.bat
```

#### 2. Install Dependencies
```bash
pip install numpy
```

*(Optional for GPU acceleration)*:
If you have an NVIDIA GPU and CUDA 12 installed, install `cupy`:
```bash
pip install cupy-cuda12x
```

---

## How to Run

### 1. Model Training
To train the tokenizer on `tiny-shakespeare.txt`, initialize model weights, and run the training loop:

```bash
python train.py
```
*(or `uv run train.py`)*

During training, weight checkpoints (`chhota_np_llm_e*.npz`) will be saved per epoch, and `tokenizer.pkl` will be exported upon completion.

### 2. Text Generation / Inference
To generate sample text using saved checkpoint weights:

```bash
python infer.py
```
*(or `uv run infer.py`)*

### 3. Model Evaluation
To evaluate trained weights and compute Cross-Entropy Loss & Perplexity (PPL) on sample text:

```bash
python eval.py
```
*(or `uv run eval.py`)*

---

## Documentation & Technical Notes

For deep-dive mathematical explanations and architectural diagrams of the forward and backward passes:
- Compile `notes.typ` into PDF format using [Typst](https://typst.app/):
  ```bash
  typst compile notes.typ
  ```
- Or view the pre-compiled `notes.pdf`.
