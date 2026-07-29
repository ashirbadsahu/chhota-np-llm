import numpy as np
from arr import as_device

class LanguageModelDataset:
    def __init__(self, texts, tokenizer, max_seq_len=512):
        self.tokenizer = tokenizer
        self.total_len = max_seq_len + 1
        self.sequences = []
        self.masks = []

        pad_id = self.tokenizer.vocab_size - 1

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) == 0:
                continue

            if len(tokens) < self.total_len:
                padded = tokens + [pad_id] * (self.total_len - len(tokens))
                m = [1] * len(tokens) + [0] * (self.total_len - len(tokens))
                self.sequences.append(padded)
                self.masks.append(m)
            else:
                for i in range(0, len(tokens), self.total_len):
                    chunk = tokens[i : i + self.total_len]
                    if len(chunk) > 1:
                        if len(chunk) < self.total_len:
                            padded = chunk + [pad_id] * (self.total_len - len(chunk))
                            m = [1] * len(chunk) + [0] * (self.total_len - len(chunk))
                            self.sequences.append(padded)
                            self.masks.append(m)
                        else:
                            self.sequences.append(chunk)
                            self.masks.append([1] * len(chunk))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = np.array(self.sequences[idx][:self.total_len], dtype=np.int32)
        mask = np.array(self.masks[idx][:self.total_len], dtype=np.float32)
        return tokens, mask


class DataLoader:
    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for batch_start in range(0, len(self.dataset), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(self.dataset))
            batch_indices = self.indices[batch_start:batch_end]

            batch = np.array([self.dataset[i][0] for i in batch_indices])
            mask = np.array([self.dataset[i][1] for i in batch_indices])

            yield as_device(batch), as_device(mask)
