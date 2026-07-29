from arr import xp

class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.W = xp.random.randn(vocab_size, embed_dim) * xp.sqrt(1.0 / vocab_size)

    def forward(self, token_indices):
        self.token_indices = token_indices
        return self.W[token_indices]

    def backward(self, grad_output):
        grad_W = xp.zeros_like(self.W)
        xp.add.at(grad_W, self.token_indices.ravel(), grad_output.reshape(-1, self.embed_dim))
        return grad_W


class PositionalEncoding:
    def __init__(self, embed_dim, max_seq_len=5000):
        self.embed_dim = embed_dim
        self.pe = xp.zeros((max_seq_len, embed_dim))

        position = xp.arange(max_seq_len).reshape(-1, 1)
        dim_indices = xp.arange(0, embed_dim, 2)
        div_term = 1.0 / (10000 ** (dim_indices / embed_dim))
        self.pe[:, 0::2] = xp.sin(position * div_term)
        if embed_dim % 2 == 1:
            self.pe[:, 1::2] = xp.cos(position * div_term[:-1])
        else:
            self.pe[:, 1::2] = xp.cos(position * div_term)

    def forward(self, x):
        seq_len = x.shape[1]
        return x + self.pe[:seq_len, :][xp.newaxis, :, :]

    def backward(self, grad_output):
        return grad_output


def gelu(x):
    return 0.5 * x * (1.0 + xp.tanh(xp.sqrt(2.0 / xp.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(x):
    x = xp.atleast_1d(x)
    t = xp.tanh(xp.sqrt(2.0 / xp.pi) * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1 - t ** 2) * xp.sqrt(2.0 / xp.pi) * (1 + 3 * 0.044715 * x ** 2) + 0.5 * (1 + t)


class FeedForwardNetwork:
    def __init__(self, embed_dim, expansion_factor=4):
        self.embed_dim = embed_dim
        self.ff_dim = embed_dim * expansion_factor

        self.W1 = xp.random.randn(embed_dim, self.ff_dim) * xp.sqrt(2.0 / embed_dim)
        self.b1 = xp.zeros((1, self.ff_dim))
        self.W2 = xp.random.randn(self.ff_dim, embed_dim) * xp.sqrt(2.0 / self.ff_dim)
        self.b2 = xp.zeros((1, embed_dim))
        self.cache = {}

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = gelu(z1)
        output = h @ self.W2 + self.b2
        self.cache = {'x': x, 'z1': z1, 'h': h}
        return output

    def backward(self, grad_output):
        x = self.cache['x']
        z1 = self.cache['z1']
        h = self.cache['h']

        batch, seq, _ = x.shape

        x_flat = x.reshape(-1, self.embed_dim)
        h_flat = h.reshape(-1, self.ff_dim)
        grad_output_flat = grad_output.reshape(-1, self.embed_dim)

        grad_W2 = h_flat.T @ grad_output_flat
        grad_b2 = xp.sum(grad_output_flat, axis=0, keepdims=True)

        grad_h = grad_output_flat @ self.W2.T

        z1_flat = z1.reshape(-1, self.ff_dim)
        gelu_g = gelu_backward(z1_flat)
        grad_z1 = grad_h * gelu_g

        grad_W1 = x_flat.T @ grad_z1
        grad_b1 = xp.sum(grad_z1, axis=0, keepdims=True)

        grad_input = grad_z1 @ self.W1.T
        grad_input = grad_input.reshape(batch, seq, self.embed_dim)

        return (
            grad_input,
            (grad_W1, grad_b1, grad_W2, grad_b2)
        )


class ScaledDotProductAttention:
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.d_k = embed_dim
        self.cache = {}

    def forward(self, Q, K, V, mask=None):
        scores = xp.matmul(Q, K.transpose(0, 1, 3, 2))
        scores = scores / xp.sqrt(self.d_k)

        if mask is not None:
            scores = scores + ((1.0 - mask) * -1e9)

        weights = self.softmax(scores)
        output = xp.matmul(weights, V)

        self.cache = {'Q': Q, 'K': K, 'V': V, 'weights': weights, 'scores': scores}
        return output, weights

    def softmax(self, x):
        e_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        return e_x / xp.sum(e_x, axis=-1, keepdims=True)

    def backward(self, grad_output):
        weights = self.cache['weights']
        V = self.cache['V']

        grad_V = xp.matmul(weights.transpose(0, 1, 3, 2), grad_output)
        grad_weights = xp.matmul(grad_output, V.transpose(0, 1, 3, 2))

        grad_scores = self.grad_softmax(weights, grad_weights)

        grad_Q = xp.matmul(grad_scores, self.cache['K'] / xp.sqrt(self.d_k))
        grad_K = xp.matmul(grad_scores.transpose(0, 1, 3, 2), self.cache['Q'] / xp.sqrt(self.d_k))

        return grad_Q, grad_K, grad_V

    def grad_softmax(self, softmax_output, grad_out):
        grad_scores = softmax_output * grad_out
        grad_scores -= softmax_output * xp.sum(grad_out * softmax_output, axis=-1, keepdims=True)
        return grad_scores


class MultiHeadAttention:
    def __init__(self, embed_dim, num_heads=8):
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.W_Q = xp.random.randn(embed_dim, embed_dim) * xp.sqrt(1.0 / embed_dim)
        self.W_K = xp.random.randn(embed_dim, embed_dim) * xp.sqrt(1.0 / embed_dim)
        self.W_V = xp.random.randn(embed_dim, embed_dim) * xp.sqrt(1.0 / embed_dim)
        self.W_O = xp.random.randn(embed_dim, embed_dim) * xp.sqrt(1.0 / embed_dim)

        self.attention = ScaledDotProductAttention(self.d_k)

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, _ = Q.shape

        self.cache_mha = {'Q_orig': Q, 'K_orig': K, 'V_orig': V}

        Q_proj = xp.dot(Q, self.W_Q).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K_proj = xp.dot(K, self.W_K).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V_proj = xp.dot(V, self.W_V).reshape(batch_size, seq_len, self.num_heads, self.d_k)

        Q_proj = Q_proj.transpose(0, 2, 1, 3)
        K_proj = K_proj.transpose(0, 2, 1, 3)
        V_proj = V_proj.transpose(0, 2, 1, 3)

        attn_output, weights = self.attention.forward(Q_proj, K_proj, V_proj, mask)

        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        output = xp.dot(attn_output, self.W_O)

        self.cache_mha['attn_output'] = attn_output
        return output, weights

    def backward(self, grad_output):
        batch_size, seq_len, _ = grad_output.shape

        grad_W_O = xp.dot(self.cache_mha['attn_output'].reshape(-1, self.embed_dim).T, grad_output.reshape(-1, self.embed_dim))
        grad_attn_output = xp.dot(grad_output, self.W_O.T)

        grad_attn_output = grad_attn_output.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        grad_Q_proj, grad_K_proj, grad_V_proj = self.attention.backward(grad_attn_output)

        grad_Q_proj = grad_Q_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_K_proj = grad_K_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)
        grad_V_proj = grad_V_proj.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.embed_dim)

        Q_orig = self.cache_mha['Q_orig']
        K_orig = self.cache_mha['K_orig']
        V_orig = self.cache_mha['V_orig']

        grad_W_Q = xp.dot(Q_orig.reshape(-1, self.embed_dim).T, grad_Q_proj.reshape(-1, self.embed_dim))
        grad_W_K = xp.dot(K_orig.reshape(-1, self.embed_dim).T, grad_K_proj.reshape(-1, self.embed_dim))
        grad_W_V = xp.dot(V_orig.reshape(-1, self.embed_dim).T, grad_V_proj.reshape(-1, self.embed_dim))

        grad_input_Q = xp.dot(grad_Q_proj, self.W_Q.T)
        grad_input_K = xp.dot(grad_K_proj, self.W_K.T)
        grad_input_V = xp.dot(grad_V_proj, self.W_V.T)

        return grad_input_Q, grad_input_K, grad_input_V, (grad_W_Q, grad_W_K, grad_W_V, grad_W_O)

    @staticmethod
    def create_causal_mask(seq_len):
        return xp.tril(xp.ones((seq_len, seq_len)))


class LayerNormalization:
    def __init__(self, embed_dim, epsilon=1e-6):
        self.embed_dim = embed_dim
        self.epsilon = epsilon

        self.gamma = xp.ones((1, 1, embed_dim))
        self.beta = xp.zeros((1, 1, embed_dim))

        self.cache = {}

    def forward(self, x):
        mean = xp.mean(x, axis=-1, keepdims=True)
        variance = xp.var(x, axis=-1, keepdims=True)

        x_norm = (x - mean) / xp.sqrt(variance + self.epsilon)
        output = self.gamma * x_norm + self.beta

        self.cache = {'x': x, 'x_norm': x_norm, 'var': variance}
        return output

    def backward(self, grad_output):
        x = self.cache['x']
        x_norm = self.cache['x_norm']
        var = self.cache['var']

        B, T, D = grad_output.shape

        grad_gamma = xp.sum(grad_output * x_norm, axis=(0, 1), keepdims=True)
        grad_beta = xp.sum(grad_output, axis=(0, 1), keepdims=True)

        grad_x_norm = grad_output * self.gamma

        sum_grad = xp.sum(grad_x_norm, axis=-1, keepdims=True)
        sum_grad_x_norm = xp.sum(grad_x_norm * x_norm, axis=-1, keepdims=True)

        grad_x = (1.0 / (D * xp.sqrt(var + self.epsilon))) * (
            D * grad_x_norm - sum_grad - x_norm * sum_grad_x_norm
        )

        return grad_x, (grad_gamma, grad_beta)


class TransformerEncoderBlock:
    def __init__(self, embed_dim, num_heads=8, ff_expansion=4):
        self.embed_dim = embed_dim
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_expansion)

        self.ln1 = LayerNormalization(embed_dim)
        self.ln2 = LayerNormalization(embed_dim)

    def forward(self, x, mask=None):
        ln1_out = self.ln1.forward(x)
        attn_output, weights = self.mha.forward(ln1_out, ln1_out, ln1_out, mask=mask)
        ln1_residual = x + attn_output

        ln2_out = self.ln2.forward(ln1_residual)
        ffn_output = self.ffn.forward(ln2_out)
        out = ln1_residual + ffn_output

        self.cache_block = {'x': x, 'ln1_out': ln1_out, 'ln1_residual': ln1_residual, 'ln2_out': ln2_out}
        return out, weights

    def backward(self, grad_output):
        grad_ffn_residual = grad_output
        grad_ffn_input = self.ffn.backward(grad_ffn_residual)
        grad_ffn_back, grads_ffn = grad_ffn_input

        grad_ln2_input = grad_ffn_back
        grad_ln2_grads = self.ln2.backward(grad_ln2_input)
        grad_ln2_back, (grad_ln2_gamma, grad_ln2_beta) = grad_ln2_grads

        grad_attn_residual = grad_output + grad_ln2_back

        grad_Q, grad_K, grad_V, grads_mha = self.mha.backward(grad_attn_residual)
        grad_ln1_input = grad_Q + grad_K + grad_V
        grad_ln1_grads = self.ln1.backward(grad_ln1_input)
        grad_ln1_back, (grad_ln1_gamma, grad_ln1_beta) = grad_ln1_grads

        grad_input = grad_attn_residual + grad_ln1_back

        return grad_input, (grad_ln1_gamma, grad_ln1_beta, grad_ln2_gamma, grad_ln2_beta, grads_ffn, grads_mha)
