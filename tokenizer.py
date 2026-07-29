"""
BBPE Tokenizer
I tried to make it somewhat production like hehe.

UTF8 -> flatten corpus -> vectorized pair counting -> np.bincount -> best merge 
-> greedy merge application -> repeat -> merge table -> inference encode/decode
"""
import numpy as np

class ByteEncoder:
    """
    UTF-8 byte encoder/decoder
    """
    @staticmethod
    def encode(text: str):
        return np.frombuffer(text.encode("utf-8"), dtype=np.uint8).astype(np.int32)
    

    @staticmethod
    def decode(byte_array):
        byte_values = []
        for x in byte_array:
            if 0 <= x <= 255:
                byte_values.append(x)
            
        return bytes(byte_values).decode(
            "utf-8",
            errors="replace"
        )


class BBPETokenizer:
    
    def __init__(
            self,
            vocab_size = 1000,
            separator = -1,
            initial_vocab = 256
    ):
        self.vocab_size = vocab_size
        self.separator = separator

        self.next_token = initial_vocab
        self.encoder = ByteEncoder()

        self.merges = []
        
        self.token_map = {
            i: bytes([i])
            for i in range(256)
        }

    # corpus building
    def _build_flattened_corpus(self, texts):
        pieces = []

        for text in texts:
            encoded = self.encoder.encode(text)
            pieces.append(encoded)
            pieces.append(np.array([self.separator], dtype=np.int32))

        return np.concatenate(pieces)

    # vectorized pair counting
    def _get_best_pair(self, corpus):
        left = corpus[:-1]
        right = corpus[1:]

        valid = ((left != self.separator) & (right != self.separator))

        left = left[valid]
        right = right[valid]

        if len(left) == 0:
            return None, 0

        max_vocab = max(self.next_token, 10000)

        pair_ids = (left.astype(np.int64) * max_vocab + right)

        counts = np.bincount(pair_ids)

        best_pair_id = counts.argmax()

        freq = int(counts[best_pair_id])

        if freq < 2:
            return None, freq

        first = (best_pair_id // max_vocab)

        second = (best_pair_id % max_vocab)

        return (int(first), int(second)), freq

    def _merge_pair(self, corpus, pair, new_token):
        a, b = pair

        left = corpus[:-1]
        right = corpus[1:]

        merge_mask = ((left == a) & (right == b))
        positions = np.where(merge_mask)[0]

        if len(positions) == 0:
            return corpus

        # Greedy selection of non-overlapping positions
        filtered_positions = []
        i = 0
        while i < len(positions):
            pos = positions[i]
            filtered_positions.append(pos)
            i += 1
            while i < len(positions) and positions[i] == pos + 1:
                i += 1
        positions = np.array(filtered_positions, dtype=np.int32)

        output = corpus.copy()

        output[positions] = new_token

        remove = np.zeros(len(corpus), dtype=bool)

        remove[positions + 1] = True

        output = output[~remove]

        return output

    # vocabulary tracking
    def _register_merge(self, pair, new_token):
        left = self.token_map[pair[0]]
        right = self.token_map[pair[1]]

        self.token_map[new_token] = left + right

    # training
    def train(self, texts, verbose=True):

        corpus = self._build_flattened_corpus(texts)

        merges_needed = (self.vocab_size - 256)

        for step in range(merges_needed):

            pair, freq = self._get_best_pair(corpus)

            if pair is None:
                if verbose:
                    print("Training stopped.")

                break

            new_token = self.next_token
            corpus = self._merge_pair(corpus, pair, new_token)
            self._register_merge(pair, new_token)
            self.merges.append((pair, new_token))
            self.next_token += 1

            if verbose:

                left = self.token_map[pair[0]]
                right = self.token_map[pair[1]]
                merged = left + right

                try:
                    readable = merged.decode("utf-8")
                except Exception:
                    readable = str(merged)

                print(f"{step+1}: {pair} -> {new_token} (freq={freq}) {readable}")

    # encoding
    def encode(self, text):
        tokens = self.encoder.encode(text).astype(np.int32)

        for pair, new_token in self.merges:
            tokens = self._merge_pair(tokens, pair, new_token)

        return tokens.tolist()

    # decoding
    def decode(self, tokens):
        bytes_out = []

        if hasattr(tokens, 'tolist'):
            tokens = tokens.tolist()
        elif hasattr(tokens, 'get'):
            tokens = tokens.get().tolist()

        for token in tokens:
            token = int(token)
            if token not in self.token_map:
                continue

            bytes_out.extend(self.token_map[token])

        return bytes(bytes_out).decode("utf-8", errors="replace")

    # diagnostics
    def vocab_length(self):
        return len(self.token_map)

    def print_vocab(self, n=20):
        shown = 0
        for k, v in self.token_map.items():
            try:
                decoded = v.decode("utf-8")
            except Exception:
                decoded = str(v)

            print(k, decoded)

            shown += 1

            if shown >= n:
                break
