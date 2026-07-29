#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge
#import fletcher.shapes: circle
#set page(
  paper: "a4",
  numbering: "1",
)
#let date = datetime(
  year: 2026,
  month: 07,
  day: 29,
)

#let day-num = date.day()
#let suffix = if calc.rem(day-num, 10) == 1 and day-num != 11 {
  "st"
} else if calc.rem(day-num, 10) == 2 and day-num != 12 {
  "nd"
} else if calc.rem(day-num, 10) == 3 and day-num != 13 {
  "rd"
} else {
  "th"
}

#align(center + horizon)[
  #title(text(size: 32pt, weight: "bold")[LLM from scratch notes])
  #v(2em)
  #text([Ashirbad Sahu], size: 12pt) \
  #str(day-num)#super(suffix) #date.display(" [month repr:long] [year]")
]




#show heading.where(level: 1): it => block(
  below: 2em,
  width: 100%,
  align(center, it)
)

#pagebreak()

#outline(title: [Table of Contents])
#pagebreak()

= LLM from scratch using numpy and cupy
This is the documentation and notes for my project to build a large language model (LLM) from scratch using numpy and cupy. The goal of this project is to understand the inner workings of LLMs by implementing one from the ground up, without relying on high-level libraries like PyTorch or TensorFlow.

== Elements of an LLM
- Tokenization
- Embeddings
- Attention Mechanism
- Transformer Architecture
- Training Loop
- Optimization
- Evaluation Metrics

#figure(
  image("images/original_transformer_architecture.png", width: 80%),
  caption: [
    Original transformer architecture
  ],
)


#pagebreak()

= Tokenization
Converting raw text into numerical representations called tokens, which can be processed by the model. Common tokenization techniques include:
- Word-level tokenization
- Subword tokenization (e.g., Byte Pair Encoding)
- Character-level tokenization
I have implemented *Bytelevel Byte Pair Encoding* (BBPE) for my LLM, which allows for efficient handling of out-of-vocabulary words and better representation of rare words.

== BBPE (Bytelevel Byte Pair Encoding)
BBPE starts with a vocabulary containing all possible byte values.

Since a byte can represent values from 0 to 255, the initial vocabulary size is:

$|V_0|$ = 256

This guarantees that any UTF-8 text can always be represented without needing an unknown token (<UNK>).

Let:
N = corpus length
M = number of merge operations
A naive implementation requires: $upright(O)("MN")$ time complexity, because pair frequencies may need recomputation after every merge. However, with efficient data structures and caching, the complexity can be reduced.

=== How BBPE works
- Start by cleaning and preprocessing the text data. Normalize whitespace, remove unwanted characters if needed, and make sure everything is UTF-8 encoded so the tokenizer can work consistently across different languages and symbols.
- Split the text into bytes instead of characters or words. Since BBPE operates directly on raw bytes, every piece of text can be represented without needing an unknown token.
- Add special markers or boundaries where required(like `</w>`). Some implementations keep track of word boundaries or whitespace so merges do not accidentally combine unrelated parts of the text.
- Flatten the entire corpus into one long sequence of byte-level tokens. At this stage, every byte is treated as an individual token in the vocabulary.
- Count all adjacent token pairs across the corpus and track how often each pair appears.
- Find the most frequent pair of consecutive tokens. This pair is assumed to represent a commonly occurring pattern in the data.
- Merge that pair into a new single token and add it to the vocabulary.
- Replace every occurrence of that pair in the corpus with the newly created merged token.
- Recalculate pair frequencies because the merges change the surrounding token combinations.
- Repeat the process: count pairs, merge the most frequent one, and update the corpus after each merge.
- Continue merging until the desired vocabulary size is reached or no useful merges remain.
- Save the final vocabulary and merge rules. These rules are later used to tokenize new text by replaying the same merge operations in the same order.

==== Example
Suppose our training corpus contains the sentence:

```text
Players played before playing
```

BBPE first converts the text into smaller units (bytes or characters for simplicity in this example).

```text
[P] [l] [a] [y] [e] [r] [s]
[p] [l] [a] [y] [e] [d]
[b] [e] [f] [o] [r] [e]
[p] [l] [a] [y] [i] [n] [g]
```

*Step 1: Count adjacent pairs*

We count how often every adjacent pair appears across the corpus.

Some frequent pairs are:

```text
(p, l) -> 2
(l, a) -> 2
(a, y) -> 2
(y, e) -> 2
```

Assume the most frequent pair selected is:

```text
(p, l) -> pl
```

*Step 2: Merge the pair*

Replace every occurrence of `(p, l)` with the new token `pl`.

```text
[P] [l] [a] [y] [e] [r] [s]
[pl] [a] [y] [e] [d]
[b] [e] [f] [o] [r] [e]
[pl] [a] [y] [i] [n] [g]
```

Now we recalculate pair frequencies.

Some updated pairs become:

```text
(pl, a) -> 2
(a, y) -> 2
```

Suppose we now merge:

```text
(pl, a) -> pla
```

*Step 3: Merge again*

After applying the merge:

```text
[P] [l] [a] [y] [e] [r] [s]
[pla] [y] [e] [d]
[b] [e] [f] [o] [r] [e]
[pla] [y] [i] [n] [g]
```

The pair `(pla, y)` also appears frequently, so we merge it next:

```text
(pla, y) -> play
```

After the merge:

```text
[P] [l] [a] [y] [e] [r] [s]
[play] [e] [d]
[b] [e] [f] [o] [r] [e]
[play] [i] [n] [g]
```

*Learned Subword Tokens*

Over time, BBPE learns commonly occurring subwords such as:

```text
pl
pla
play
ing
ed
```

So instead of storing every word independently, the tokenizer can represent words using reusable subword units.

For example:

```text
played  -> [play] [ed]
playing -> [play] [ing]
```

This is what makes BBPE efficient: it learns reusable patterns that appear across many different words.

#pagebreak()
= Positional Encoding

== Why do we need positional encoding?
Transformers use self attention, which is permutation-invariant (order insensitive). Without positional information, the model cannot distinguish between sequences with the same
tokens in different orders.

=== Example:
- *Sentence A:* "Only *he* wants to eat biryani." (No one else wants to eat it)
- *Sentence B:* "*He* wants to eat only biryani." (He wants to eat nothing else)

==== The Problem:
Both sentences contain the exact same tokens: `{"only", "he", "wants", "to", "eat", "biryani"}`. 

Without positional encoding, the Transformer cannot determine which word "only" is modifying. It generates identical mathematical representations for both sentences, completely failing to capture the difference in meaning.

==== The Solution:
Positional encoding adds a unique structural "coordinate" to each word vector before it enters the attention mechanism:

- In Sentence A, "only" is tagged at *Position 1* (attending closely to "he").
- In Sentence B, "only" is tagged at *Position 5* (attending closely to "biryani").

This extra mathematical information breaks the order-insensitivity, allowing the model to correctly interpret the syntax and context.
== Math of Positional Encoding in the Project

In this project, a deterministic *sinusoidal* positional embedding is used to inject sequence order into the model.

- *For even dimension indices ($2i$):* $ "PE"_("pos", 2 i) = sin("pos" / (10000^((2 i) / d_e))) $
- *For odd dimension indices ($2i + 1$):* $ "PE"_("pos", 2 i + 1) = cos("pos" / (10000^((2 i) / d_e))) $

Where:
- $"pos"$: The absolute token position in the sequence ($0$ to $T-1$).
- $i$: The frequency dimension index ($0$ to $d_e / 2 - 1$).
- $d_e$: The total embedding dimension.


=== Capturing Relative Distance via Linear Combination

The core genius of the sine-cosine formulation is that it allows the model to easily learn _relative_ distances. For any fixed token offset $k$, the positional encoding at position $"pos" + k$ can be represented as a *linear transformation (rotation)* of the encoding at position $"pos"$.

For a specific frequency component $omega_i = 1 / 10000^((2i)/d_e)$, we can pair the even (sine) and odd (cosine) dimensions into a 2D vector space. Using the trigonometric angle addition identities:

$ sin(omega_i ("pos" + k)) = sin(omega_i "pos")cos(omega_i k) + cos(omega_i "pos")sin(omega_i k) $
$ cos(omega_i ("pos" + k)) = cos(omega_i "pos")cos(omega_i k) - sin(omega_i "pos")sin(omega_i k) $

This can be elegantly refactored into a matrix multiplication (a linear combination):

$
vec("PE"_("pos"+k, 2i), "PE"_("pos"+k, 2i+1)) = 
mat(
  cos(omega_i k), sin(omega_i k);
  -sin(omega_i k), cos(omega_i k)
) 
vec("PE"_("pos", 2i), "PE"_("pos", 2i+1))
$

==== Why this matters:
1. *Position-Independent Transformation:* The $2 times 2$ rotation matrix depends *only* on the relative offset $k$, not on the absolute position $"pos"$. 
2. *Attention Mechanism Compatibility:* When the self-attention layer calculates dot-products ($Q^T K$) between two tokens separated by distance $k$, the resulting attention score is a direct function of this relative rotation matrix. This enables the model to easily generalize to tracking how far apart words are from each other regardless of where they appear in a sentence.

#pagebreak()

= Feed Forward Networks (FFNs)

== What is a Feed Forward Network?
An FFN (Feed-Forward Neural Network) is a standard layer of artificial neurons where data moves in one direction only (forward, from input to output through hidden layers).

- Mathematically:
$ "FFN"(x) = "Activation"(x W_1 + b_1) W_2 + b_2 $

Where:
- $x$: Input vector
- $W_1$: First weight matrix, projects $x$ into a much higher-dimensional space
- $b_1$: First bias vector
- $"Activation"()$: Non-linear activation function
- $W_2$: Second weight matrix, compresses expanded features back to model's standard size
- $b_2$: Second bias vector
- $"FFN"(x)$: Output vector

== Activation Functions
- Activation functions are applied to every number in the vector. They decide which values pass through and which get muted or zeroed out, allowing the network to learn complex, non-linear relationships rather than simple linear algebra.
- Common activation functions include GELU, SwiGLU, ReLU, Sigmoid, and Tanh.
- In this project, *GELU (Gaussian Error Linear Unit)* is used.

=== Gaussian Error Linear Unit (GELU)
GELU scales inputs by their probability under a Gaussian distribution, providing a smooth, non-linear transformation that allows small negative gradients to flow.

- Exact Definition:
$ "GELU"(x) = x · Phi(x) = x · 1/2 [1 + "erf"(x / sqrt(2))] $

Where:
- $Phi(x)$: Cumulative Distribution Function (CDF) of standard normal distribution
- $"erf"(x)$: Error function calculating area under a bell curve

- Fast Approximation:
For computational efficiency, frameworks commonly approximate GELU as:
$ "GELU"(x) approx 0.5x · (1 + "tanh"(sqrt(2 / pi) (x + 0.044715 x^3))) $

#pagebreak()
= Attention Mechanism
Attention allows the model to focus on relevant parts of the input sequence when processing each token by assigning different weights to different tokens.

== Mathematical Definition
Given:
- Query matrix $Q in RR^(T times d_k)$ (what we are looking for)
- Key matrix $K in RR^(T times d_k)$ (what we match against)
- Value matrix $V in RR^(T times d_v)$ (what we retrieve), where $d_k$ and $d_v$ are the key and value dimensions respectively

Attention is computed as:
$ bold("Attention")(Q, K, V) = "softmax"((Q K^T) / sqrt(d_k)) V $

=== Step-by-Step Breakdown:
1. *Compute similarity scores:*
   $ "scores" = Q K^T in RR^(T times T) $
   Measures how much each token in the sequence relates to every other token.

2. *Scale scores:*
   $ "scaled_scores" = (Q K^T) / sqrt(d_k) $
   Scales down the dot products by $sqrt(d_k)$ to prevent extremely large values, which cause vanishing gradients in the softmax function.

3. *Apply Softmax:*
   $ "weights" = "softmax"("scaled_scores") $
   Normalizes the scores across each row into probabilities between $0$ and $1$ that sum to $1$.

4. *Weight the Values:*
   $ "output" = "weights" · V in RR^(T times d_v) $
   Computes a weighted sum of the Value vectors using the softmax attention probabilities to form the final representation.

=== Intuitive Example

Consider a simple three-word sentence: *"The cat sat"*

For these three words, the model projects Query ($bold(Q)$), Key ($bold(K)$), and Value ($bold(V)$) matrices:

$
bold(Q) = [bold(q)_("The"), bold(q)_("cat"), bold(q)_("sat")] \
bold(K) = [bold(k)_("The"), bold(k)_("cat"), bold(k)_("sat")] \
bold(V) = [bold(v)_("The"), bold(v)_("cat"), bold(v)_("sat")]
$

When the model is specifically *processing the word "sat"*, it takes the query vector $bold(q)_("sat")$ and compares it against all key vectors ($bold(k)$) using dot products to measure relevance:

- $bold(q)_("sat") dot bold(k)_("cat")$ is *high*: "Sat" (an action) strongly relates to "cat" (the noun performing the action).
- $bold(q)_("sat") dot bold(k)_("The")$ is *medium*: "The" provides minor grammatical context.
- $bold(q)_("sat") dot bold(k)_("sat")$ is *low*: "Sat" doesn't need to look at itself as much to gain extra meaning here.

*1. Softmax converts these similarity scores into attention weights (percentages):*
$ "softmax"(["medium", "high", "low"]) -> [0.2, 0.7, 0.1] $

*2. Constructing the final output:*
The model uses these weights to combine the information vectors ($bold(v)$):
$ "Output" = 0.2 bold(v)_("The") + 0.7 bold(v)_("cat") + 0.1 bold(v)_("sat") $

*Result:* The final representation for "sat" now carries *mostly information about the "cat"*, giving the model the context that *a cat is the thing sitting*.

== Multi-Head Attention

Instead of performing attention once across the entire vector dimension, *Multi-Head Attention* splits the Query, Key, and Value representations into $h$ smaller "heads". This allows the model to jointly attend to information from different representation subspaces at different positions.

=== Mathematical Definition

Given number of heads $h$, and projection matrices $W_i^Q, W_i^K in RR^(d_("model") times d_k)$, $W_i^V in RR^(d_("model") times d_v)$, and $W^O in RR^(h d_v times d_("model"))$:

$ "MultiHead"(Q, K, V) = "Concat"("head"_1, "head"_2, ..., "head"_h) W^O $

Where each individual head $i$ is calculated as:
$ "head"_i = "Attention"(Q W_i^Q, K W_i^K, V W_i^V) $

=== Step-by-Step Breakdown

1. *Linear Projections:* Project input matrices $Q$, $K$, and $V$ into $h$ lower-dimensional subspaces using weight matrices $W_i^Q$, $W_i^K$, and $W_i^V$.
2. *Parallel Attention:* Compute scaled dot-product attention for each head independently in parallel.
3. *Concatenation:* Concatenate the outputs from all $h$ heads along the feature dimension.
4. *Final Output Projection:* Multiply the concatenated result by the output projection matrix $W^O$ to restore the vector back to $d_("model")$.

=== Intuitive Example

Consider the sentence: *"The animal didn't cross the street because it was too tired."*

A single attention head might struggle to decide what *"it"* refers to. Multi-Head Attention solves this by letting different heads focus on different relationships simultaneously:

- *Head 1:* Connects *"it"* to *"animal"* (identifying the noun coreference).
- *Head 2:* Connects *"it"* to *"tired"* (identifying the state/property).
- *Head 3:* Connects *"cross"* to *"street"* (identifying verb-object structure).

By combining all heads, the model builds a rich, multi-dimensional understanding of the token *"it"*.


== Causal Masking (Lower Triangular Mask)

In auto-regressive models (like GPT), predicting the next token requires that a position *cannot look ahead* to future tokens during training. *Causal Masking* enforces this by blocking attention scores between current tokens and future positions.

=== Mathematical Definition

Before applying Softmax, a mask matrix $M$ is added to the scaled query-key scores:

$ "CausalAttention"(Q, K, V) = "softmax"(((Q K^T) / sqrt(d_k)) + M) V $

Where the mask matrix $M in RR^(T times T)$ is defined as:

$ M_(i, j) = cases(
  0 & "if" i >= j,
  -infinity & "if" i < j
) $

=== Step-by-Step Breakdown

1. *Compute Raw Scores:* Calculate the attention matrix $S = (Q K^T) / sqrt(d_k)$ of size $T times T$.
2. *Apply Mask $M$:* Add $-infinity$ to all elements above the main diagonal (where column index $j >$ row index $i$).
3. *Apply Softmax:* Because $e^(-infinity) = 0$, the attention weights for all future positions become strictly $0$.
4. *Weighted Output:* Output for position $i$ becomes a weighted sum strictly over tokens at positions $j <= i$.

=== Intuitive Example

For the sentence: *"The cat sat"* ($T = 3$)

Without masking, the score matrix allows every word to attend to every other word:

$
S = [
  "The" -> "The", "The" -> "cat", "The" -> "sat";
  "cat" -> "The", "cat" -> "cat", "cat" -> "sat";
  "sat" -> "The", "sat" -> "cat", "sat" -> "sat"
]
$

Adding the causal mask matrix $M$:

$
M = [
  0, -infinity, -infinity;
  0, 0, -infinity;
  0, 0, 0
]
$

After applying Softmax, the resulting attention weight matrix becomes:

$
"Weights" = [
  1.0, 0.0, 0.0;
  0.3, 0.7, 0.0;
  0.2, 0.7, 0.1
]
$

- Row 1 (*"The"*): Can only look at *"The"* ($100\%$). Future words *"cat"* and *"sat"* are masked out to $0\%$.
- Row 2 (*"cat"*): Can look at *"The"* ($30\%$) and *"cat"* ($70\%$). Future word *"sat"* is masked out to $0\%$.
- Row 3 (*"sat"*): Can look at all previous words (*"The"*, *"cat"*, *"sat"*) because no future words remain.



#pagebreak()

= Transformer Block: Layer Normalization & Residual Connections

== 1. Layer Normalization (LayerNorm)

=== What is Convergence?
*Convergence* means reaching a state where training stabilizes because the loss curve flattens out, indicating the model has minimized its error as much as it reasonably can.

=== Why Do We Need LayerNorm?
In deep Transformer networks, signal activations pass through dozens of stacked layers. Small variations at the input compound into massive spikes or vanishing signals in deeper layers (termed *internal covariate shift*). LayerNorm rescales activations across the feature dimension for each token independently, stabilizing signal flow and enabling faster, more stable convergence.

=== Mathematical Definition
For an input tensor $x in RR^(T times d_("model"))$, LayerNorm operates across the feature dimension $d_("model")$ for each individual token vector $x_t \in RR^(d_("model"))$:

$ "LayerNorm"(x_t) = gamma dot.o (frac(x_t - mu, sqrt(sigma^2 + epsilon))) + beta $

Where:
- $x_t$: Input feature vector for a single token at position $t$ of shape $(d_("model"))$.
- $mu$: Mean of the token's features, $mu = 1/d_("model") sum_(i=1)^(d_("model")) x_(t,i)$ (Scalar).
- $sigma^2$: Variance of the token's features, $sigma^2 = 1/d_("model") sum_(i=1)^(d_("model")) (x_(t,i) - mu)^2$ (Scalar).
- $epsilon$: Tiny constant (e.g., $10^(-5)$) added to avoid division by zero when variance is tiny.
- $(x_t - mu) / sqrt(sigma^2 + epsilon)$: *Standardization step*, it  forces features to have zero mean ($mu=0$) and unit variance ($sigma^2=1$).
- $gamma$: Learnable *scaling parameter* of shape $(d_("model"))$. Restores expressive power if pure unit variance is sub-optimal.
- $beta$: Learnable *shift parameter* (bias) of shape $(d_("model"))$. Restores model expressivity by allowing shifted centers.

=== Key Difference: LayerNorm vs. BatchNorm

#table(
  columns: (1fr, 2fr, 2fr),
  stroke: 0.5pt + luma(150),
  fill: (x, y) => if y == 0 { rgb("#e8edf2") } else { none },
  [*Metric*], [*Batch Normalization (BatchNorm)*], [*Layer Normalization (LayerNorm)*],
  [*Normalization Axis*], [Across the *batch* dimension ($N$) for each feature.], [Across the *feature* dimension ($d_("model")$) for each token.],
  [*Batch Dependency*], [High. Small or variable batch sizes break stability.], [None. Works identically for batch size = 1 or variable lengths.],
  [*Primary Use Case*], [Computer Vision / Convolutional Networks.], [Sequence Models / Transformers / LLMs.]
)

#v(1em)

== 2. Residual Connections (Skip Connections)

=== Why Do We Need Residual Connections?
In deep neural networks, passing data sequentially through dozens of layers (multiplications by weights + non-linear activations like GELU) degrades the signal. Information gets lost, and during backpropagation, the training signals (gradients) vanish as they are repeatedly multiplied by small weight matrices.

Residual connections bypass a sub-layer by adding its original input directly to its output ($y = x + F(x)$). This creates an "information highway" that preserves both raw features and gradient signals.

=== Standard Transformer Block Architecture (Pre-LN)

Modern LLMs (like LLaMA and GPT-3) use *Pre-Layer Normalization*, where LayerNorm is applied _before_ the sub-layer function inside the residual connection:

1. *Multi-Head Attention Sub-Layer:*
   $ x_("attn") = x + "MultiHeadAttention"("LayerNorm"(x)) $

2. *Feed-Forward Network (FFN) Sub-Layer:*
   $ x_("out") = x_("attn") + "FFN"("LayerNorm"(x_("attn"))) $

#v(0.5em)

// Reusable sub-layer inner block diagram
#figure(
  caption: [Pre-LN Residual Connection with GELU Feed-Forward Sub-Layer],
  numbering: none,
  diagram(
    spacing: (10mm, 8mm),
    node-stroke: 0.8pt + rgb("#2b2b2b"),
    edge-stroke: 1pt + rgb("#2b2b2b"),
    node-corner-radius: 4pt,

    // Nodes
    node((0, 0), [$x$], stroke: none),
    node((1, 0), name: <branch>, stroke: none),
    node((2, 0), name: <ln>, fill: rgb("#deebf7"), [LayerNorm]),
    node((3, 0), name: <w1>, fill: rgb("#e2f0d9"), [Linear\ $W_1$]),
    node((4, 0), name: <gelu>, fill: rgb("#fff2cc"), [GELU]),
    node((5, 0), name: <w2>, fill: rgb("#e2f0d9"), [Linear\ $W_2$]),
    node((6, 0), name: <add>, shape: circle, radius: 2.5mm, fill: rgb("#fce4d6"), [$+$]),
    node((7, 0), name: <out>, stroke: none, [$x + F(x)$]),

    // Main Flow Edges
    edge((0, 0), <branch>, "-|>"),
    edge(<branch>, <ln>, "-|>"),
    edge(<ln>, <w1>, "-|>"),
    edge(<w1>, <gelu>, "-|>"),
    edge(<gelu>, <w2>, "-|>", label: [$F(x)$], label-side: right),
    edge(<w2>, <add>, "-|>"),
    edge(<add>, <out>, "-|>"),

    // Skip Connection
    edge(
      <branch>, <add>,
      "-|>",
      bend: -45deg,
      label: [Identity Shortcut ($x$)],
      label-side: left,
      stroke: 1.2pt + rgb("#1f4e78")
    ),
  )
)

=== Mathematical Proof: Solving Vanishing Gradients

Consider a deep network with $L$ residual layers. The output at final layer $L$ is a direct recursive sum of earlier inputs and sub-layer transformations $f_i$:

$ x_L = x_0 + sum_(i=0)^(L-1) f_i (x_i) $

When computing the gradient of Loss ($L$) with respect to early activations $x_0$, we apply the chain rule:

$ frac(partial cal(L), partial x_0) = frac(partial cal(L), partial x_L) dot frac(partial x_L, partial x_0) = frac(partial cal(L), partial x_L) dot (1 + frac(partial, partial x_0) sum_(i=0)^(L-1) f_i (x_i)) $

#rect(fill: rgb("#f4f6f8"), inset: 10pt, radius: 4pt, width: 100%)[
  *Key Insight:* 
  Notice the *"$1$"* inside the bracket. Even if the transformed gradient term $frac(partial, partial x_0) sum f_i (x_i)$ approaches zero (vanishes), the gradient simplifies to:
  
  $ frac(partial cal(L), partial x_0) approx frac(partial cal(L), partial x_L) dot (1) $
  
  This guarantees an unattenuated, direct path for training gradients to flow from the very last layer back to the input parameters without decaying to zero!
]
== Transformer Blocks: Encoder vs. Decoder
#v(1em)
=== Transformer Encoder Block

An Encoder block processes the entire input sequence simultaneously with bi-directional self-attention. It uses a *Pre-LayerNorm* architecture with two main sub-layers: Multi-Head Self-Attention (MHSA) and a Position-wise Feed-Forward Network (FFN).

==== Forward Pass Mathematics

Given input sequence embeddings $x_0 \in RR^(T times d_("model"))$:

1. *Sub-Layer 1: Multi-Head Self-Attention*
   $ x_1 = x_0 + "MultiHeadAttention"("LayerNorm"(x_0)) $

2. *Sub-Layer 2: Position-wise Feed-Forward Network*
   $ x_2 = x_1 + "FFN"("LayerNorm"(x_1)) $

*Output:* Encoded representation matrix $x_2 \in RR^(T times d_("model"))$.

#v(1em)

=== Transformer Decoder Block

A full Encoder-Decoder style Transformer Decoder block consists of *three* sub-layers:
1. *Masked Multi-Head Self-Attention* (prevents attending to future tokens).
2. *Encoder-Decoder Cross-Attention* (attends to encoder outputs).
3. *Position-wise Feed-Forward Network (FFN)*.

==== Forward Pass Mathematics

Given target sequence input $y_0 \in RR^(T_("tgt") times d_("model"))$ and Encoder Output $H_("enc") \in RR^(T_("src") times d_("model"))$:

1. *Sub-Layer 1: Masked Self-Attention (Causal)*
   $ y_1 = y_0 + "MultiHeadAttention"("LayerNorm"(y_0), "mask"="Causal") $
   #text(size: 8.5pt, fill: luma(100))[
     *Note:* The causal mask sets future attention weights to $-infinity$ so position $t$ cannot attend to tokens at $t+1, t+2, dots$
   ]

2. *Sub-Layer 2: Encoder-Decoder Cross-Attention*
   $ y_2 = y_1 + "CrossAttention"(Q="LayerNorm"(y_1), K=H_("enc"), V=H_("enc")) $
   #text(size: 8.5pt, fill: luma(100))[
     *Note:* Query ($Q$) comes from the decoder state ($y_1$), while Key ($K$) and Value ($V$) come from the Encoder Output ($H_("enc")$).
   ]

3. *Sub-Layer 3: Position-wise Feed-Forward Network*
   $ y_3 = y_2 + "FFN"("LayerNorm"(y_2)) $

*Output:* Decoded representation matrix $y_3 \in RR^(T_("tgt") times d_("model"))$.

#v(1em)

- _LLMs have decoder only architecture_.
=== 3. Architectural Comparison

#table(
  columns: (1fr, 2fr, 2fr, 2fr),
  stroke: 0.5pt + luma(150),
  fill: (x, y) => if y == 0 { rgb("#e8edf2") } else { none },
  [*Block Type*], [*Attention Types Used*], [*Causal Masking?*], [*Example Models*],
  [*Encoder Only*], [Bi-directional Self-Attention], [No], [BERT, RoBERTa, ViT],
  [*Decoder Only*], [Masked Causal Self-Attention], [Yes], [GPT-4, LLaMA, Claude],
  [*Encoder-Decoder*], [Causal Self-Attention + Cross-Attention], [Yes (Self-Attn only)], [T5, BART, Whisper]
)

#pagebreak()
= Complete Model Architecture and Training Loop
Modern LLMs like GPT use a decoder-only architecture. It consists of:
  - Stack of decoder blocks:
    - Input: token embedding + positional encoding
    - Masked self-attention (causal mask)
    - Feed-Forward Network (FFN)
    - Output: Probability distribution over vocabulary
  - Causal masked self-attention only (no cross-attention)
  - Simpler, more efficient training for autoregressive language modeling
== Mathematical pipeline:
$
x_"embed" = "Embed"(x) + "PE" \
h^(0) = x_"embed" \
h^(l) = "DecoderBlock"^(l)(h^(l-1)) quad "for " l = 1, ..., N \
"logits" = h^(N) dot upright(W)_"output" \
"probs" = "softmax"("logits")
$
== Cross-Entropy Loss

Cross-entropy loss measures how far the model's predicted probability distribution deviates from the actual ground-truth next token.

$
cal(L) = - 1 / (B T) sum_(b=1)^B sum_(t=1)^T log(p_b (y_(b, t) | y_(b, 1), ..., y_(b, t-1)))
$

where $cal(L)$ is the average cross-entropy loss, $p_b(y_(b, t) | ...)$ is the predicted probability assigned to the target token $y_(b, t)$ at sequence position $t$ in batch sample $b$, $B$ is the batch size, and $T$ is the sequence length.

In terms of raw logits $z$, the loss for a single token at position $(b, t)$ with true class index $k = y_(b, t)$ is:

$
cal(L)_(b, t) = -z_(b, t, k) + log(sum_(j=1)^(|V|) exp(z_(b, t, j)))
$

where $|V|$ is the vocabulary size and $z_(b, t, j)$ is the logit for token $j$.

Directly evaluating the exponential terms causes numerical overflow for large positive logits or underflow for large negative logits. To prevent this, frameworks use the *LogSumExp trick* by subtracting $m = max_j(z_j)$:

$
"LSE"(z) = m + log(sum_(j=1)^(|V|) exp(z_j - m))
$

The numerically stable loss for a single token simplifies to:

$
cal(L)_(b, t) = -z_(b, t, k) + "LSE"(z_(b, t))
$

== Adam Optimizer

Instead of vanilla Stochastic Gradient Descent (SGD), which applies a single, fixed learning rate across all parameters regardless of their update frequency, the *Adam* (Adaptive Moment Estimation) optimizer dynamically adapts learning rates per parameter. It achieves this by combining two distinct mechanics:

1. *Momentum (1st Moment):* Keeps track of an exponentially decaying average of past gradients ($m_t$) to smooth out noise, maintain directional velocity, and accelerate convergence through flat regions.
2. *RMSProp (2nd Moment):* Keeps track of an exponentially decaying average of past *squared* gradients ($v_t$) to adaptively scale learning rates per parameter. It dampens step sizes for parameters with large, frequent gradients while boosting step sizes for parameters with small or sparse gradients.

=== Mathematical Formulation

For a given time step $t$, parameter vector $theta_t$, loss function $cal(L)(theta)$, and base learning rate $eta$:

1. *Compute Gradient:*
$
g_t = nabla_(theta) cal(L)(theta_(t-1))
$

2. *Update Exponential Moving Averages (Biased Moments):*
$
m_t = beta_1 m_(t-1) + (1 - beta_1) g_t \
v_t = beta_2 v_(t-1) + (1 - beta_2) g_t^2
$

3. *Compute Bias Corrections:*
Since $m_0 = 0$ and $v_0 = 0$, both moments are heavily biased toward zero during initial steps. We correct them via:
$
hat(m)_t = m_t / (1 - beta_1^t) \
hat(v)_t = v_t / (1 - beta_2^t)
$

4. *Apply Parameter Update:*
$
theta_t = theta_(t-1) - eta / (sqrt(hat(v)_t) + epsilon) dot hat(m)_t
$

Where:
- $theta_t$: Vector of trainable model weights/parameters at step $t$.
- $g_t$: Gradient vector of the loss with respect to parameters at step $t$.
- $m_t, v_t$: Biased estimates of the $1"st"$ moment (mean gradient) and $2"nd"$ moment (uncentered variance of gradients), respectively.
- $hat(m)_t, hat(v)_t$: Unbiased (bias-corrected) $1"st"$ and $2"nd"$ moment estimates.
- $eta$: Base learning rate hyperparameter (typically $10^(-3)$ to $10^(-4)$ for transformers).
- $beta_1$: Decay rate for the $1"st"$ moment estimate (standard default: $0.9$).
- $beta_2$: Decay rate for the $2"nd"$ moment estimate (standard default: $0.999$).
- $beta_1^t, beta_2^t$: Hyperparameters raised to the power of the current step number $t$ (controlling the decay of bias correction over time).
- $epsilon$: Small constant added for numerical stability to prevent division by zero (standard default: $10^(-8)$).

#pagebreak()

= Data Pipeline and Training at Scale

== Gradient Accumulation

When memory limits prevent fitting a large target batch size into GPU VRAM, *gradient accumulation* simulates a larger effective batch size. It computes gradients across $N_"accum"$ smaller micro-batches and sums them up before calling the optimizer.

=== Algorithm

1. *Accumulate Gradients over Micro-Batches:*
$
g_"accum" = sum_(k=1)^(N_"accum") g_(k)
$

2. *Average and Update Weights:*
$
theta_t = "OptimizerUpdate"(theta_(t-1), 1 / N_"accum" g_"accum")
$

This yields an effective batch size of:
$
B_"eff" = B_"micro" times N_"accum"
$

where $B_"micro"$ is the per-device batch size per step, and $g_k$ is the gradient computed on micro-batch $k$.

== Learning Rate Scheduling

A learning rate schedule dynamically adjusts the global learning rate $eta_t$ throughout training to improve numerical stability during early iterations and refine convergence near the end.

=== Warmup Phase
Gradually ramps up the learning rate linearly from near zero to $eta_"max"$ over $T_"warm"$ steps. This prevents large, destabilizing gradient steps when parameter weights are randomly initialized or unaligned.

$
eta_t = eta_"max" dot min(t / T_"warm", 1) quad "for " 1 <= t <= T_"warm"
$

=== Cosine Decay Phase
After warmup, reduces the learning rate to a minimum baseline $eta_"min"$ along a smooth cosine curve over the remaining steps $T_"max"$. This allows the optimizer to make fine-grained parameter adjustments as it approaches a local minimum.

$
eta_t = eta_"min" + 1/2 (eta_"max" - eta_"min") dot (1 + cos(pi (t - T_"warm") / (T_"max" - T_"warm"))) quad "for " t > T_"warm"
$

=== Symbol Definitions
- $eta_t$: Global learning rate at step $t$.
- $eta_"max"$: Peak target learning rate after warmup.
- $eta_"min"$: Minimum baseline learning rate (often $0.1 dot eta_"max"$ or $0$).
- $T_"warm"$: Total number of warmup steps.
- $T_"max"$: Total number of training steps.

== Evaluation Metrics

=== Perplexity (PPL)

- Perplexity is the standard intrinsic metric for language models. It quantifies model uncertainty by measuring how "surprised" the model is when predicting the next token across a test corpus of length $N$.

- Mathematically, perplexity is the exponential of the average cross-entropy loss:

$
"PPL" = exp(cal(L)) = exp(- 1/N sum_(i=1)^N log p(y_i | y_1, ..., y_(i-1)))
$

* *Interpretation:* A perplexity of $K$ means the model is, on average, as uncertain as if it were choosing uniformly at random among $K$ equiprobable candidate tokens.
* *Lower is Better:* Lower perplexity indicates higher probability assigned to the ground-truth text sequence.