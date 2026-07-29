import numpy as np
from arr import xp, as_numpy, as_device
from layers import *

class GPTDecoder:
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_expansion=4, max_seq_len=2048):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embed = Embedding(vocab_size, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, max_seq_len)

        self.layers = [
            TransformerEncoderBlock(embed_dim, num_heads, ff_expansion) for _ in range(num_layers)
        ]

        self.ln_final = LayerNormalization(embed_dim)
        self.output_proj = xp.random.randn(embed_dim, vocab_size) * xp.sqrt(1.0 / embed_dim)

    def forward(self, token_ids):
        token_ids = as_device(token_ids)
        batch_size, seq_len = token_ids.shape

        emb = self.token_embed.forward(token_ids)
        x = self.pos_encode.forward(emb)

        causal_mask = MultiHeadAttention.create_causal_mask(seq_len)

        for layer in self.layers:
            x, _ = layer.forward(x, mask=causal_mask)

        x_norm = self.ln_final.forward(x)

        logits = xp.dot(x_norm, self.output_proj)

        self.cache_model = {'token_ids': token_ids, 'x_norm': x_norm}
        return logits

    def backward(self, grad_logits):
        token_ids = self.cache_model['token_ids']
        x_norm = self.cache_model['x_norm']

        grad_output_proj = xp.dot(x_norm.reshape(-1, self.embed_dim).T, grad_logits.reshape(-1, self.vocab_size))
        grad_x_norm = xp.dot(grad_logits.reshape(-1, self.vocab_size), self.output_proj.T).reshape(grad_logits.shape[0], grad_logits.shape[1], self.embed_dim)

        grad_x = self.ln_final.backward(grad_x_norm)[0]
        grad_ln_final_gamma, grad_ln_final_beta = self.ln_final.backward(grad_x_norm)[1]

        grads = {}
        grads['output_proj'] = grad_output_proj
        grads['ln_final_gamma'] = grad_ln_final_gamma
        grads['ln_final_beta'] = grad_ln_final_beta

        for i, layer in reversed(list(enumerate(self.layers))):
            grad_x, grads_block = layer.backward(grad_x)

            grad_ln1_gamma, grad_ln1_beta, grad_ln2_gamma, grad_ln2_beta, grads_ffn, grads_mha = grads_block

            grads[f'layer_{i}_ln1_gamma'] = grad_ln1_gamma
            grads[f'layer_{i}_ln1_beta'] = grad_ln1_beta
            grads[f'layer_{i}_ln2_gamma'] = grad_ln2_gamma
            grads[f'layer_{i}_ln2_beta'] = grad_ln2_beta

            grads[f'layer_{i}_ffn_W1'], grads[f'layer_{i}_ffn_b1'], grads[f'layer_{i}_ffn_W2'], grads[f'layer_{i}_ffn_b2'] = grads_ffn
            grads[f'layer_{i}_mha_W_Q'], grads[f'layer_{i}_mha_W_K'], grads[f'layer_{i}_mha_W_V'], grads[f'layer_{i}_mha_W_O'] = grads_mha

        grad_emb = self.pos_encode.backward(grad_x)

        grad_token_embed = self.token_embed.backward(grad_emb)
        grads['token_embed_W'] = grad_token_embed

        return grads

    def get_params(self):
        params = []
        params.append(('output_proj', self.output_proj))
        params.append(('ln_final_gamma', self.ln_final.gamma))
        params.append(('ln_final_beta', self.ln_final.beta))
        params.append(('token_embed_W', self.token_embed.W))

        for i, layer in enumerate(self.layers):
            params.append((f'layer_{i}_ln1_gamma', layer.ln1.gamma))
            params.append((f'layer_{i}_ln1_beta', layer.ln1.beta))
            params.append((f'layer_{i}_ln2_gamma', layer.ln2.gamma))
            params.append((f'layer_{i}_ln2_beta', layer.ln2.beta))

            params.append((f'layer_{i}_ffn_W1', layer.ffn.W1))
            params.append((f'layer_{i}_ffn_b1', layer.ffn.b1))
            params.append((f'layer_{i}_ffn_W2', layer.ffn.W2))
            params.append((f'layer_{i}_ffn_b2', layer.ffn.b2))

            params.append((f'layer_{i}_mha_W_Q', layer.mha.W_Q))
            params.append((f'layer_{i}_mha_W_K', layer.mha.W_K))
            params.append((f'layer_{i}_mha_W_V', layer.mha.W_V))
            params.append((f'layer_{i}_mha_W_O', layer.mha.W_O))

        return params

    def save_weights(self, path):
        params_dict = {name: as_numpy(param) for name, param in self.get_params()}
        np.savez(path, **params_dict)

    def load_weights(self, path):
        loaded = np.load(path)
        params_dict = {name: param for name, param in self.get_params()}
        for name in params_dict.keys():
            if name in loaded:
                params_dict[name][:] = as_device(loaded[name])
            else:
                print(f"Warning: Parameter {name} not found in {path}")

    def generate(self, start_tokens, max_length, temperature=1.0):
        current_tokens = as_device(start_tokens).copy()
        max_context = self.pos_encode.pe.shape[0]

        for _ in range(max_length):
            context = current_tokens[:, -max_context:]
            logits = self.forward(context)

            next_logits = logits[:, -1, :] / temperature

            probs = self.softmax(next_logits)
            next_token = xp.argmax(probs, axis=-1)

            current_tokens = xp.concatenate(
                [current_tokens, next_token.reshape(-1, 1)],
                axis=1
            )

            if int(next_token[0]) == self.vocab_size - 1:
                break

        return current_tokens

    def softmax(self, x):
        e_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        return e_x / xp.sum(e_x, axis=-1, keepdims=True)
