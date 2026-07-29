from arr import xp

class CrossEntropyLoss:
    def __init__(self):
        self.cache = {}

    def forward(self, logits, targets, mask=None):
        batch_size, seq_len, vocab_size = logits.shape

        logits_max = xp.max(logits, axis=-1, keepdims=True)
        logits_exp = xp.exp(logits - logits_max)

        log_sum_exp = logits_max.squeeze(-1) + xp.log(xp.sum(logits_exp, axis=-1))

        correct_logits = xp.take_along_axis(logits, targets[..., xp.newaxis], axis=-1).squeeze(-1)
        per_token_loss = log_sum_exp - correct_logits

        if mask is not None:
            per_token_loss = per_token_loss * mask
            loss = xp.sum(per_token_loss) / (xp.sum(mask) + 1e-8)
        else:
            loss = xp.sum(per_token_loss) / (batch_size * seq_len)

        probs = logits_exp / xp.sum(logits_exp, axis=-1, keepdims=True)
        targets_onehot = xp.eye(vocab_size)[targets]
        self.cache = {'probs': probs, 'targets_onehot': targets_onehot, 'mask': mask, 'batch_size': batch_size, 'seq_len': seq_len}

        return loss

    def backward(self):
        grad_logits = self.cache['probs'] - self.cache['targets_onehot']

        mask = self.cache['mask']
        if mask is not None:
            divisor = xp.sum(mask) + 1e-8
            grad_logits *= mask[..., xp.newaxis] / divisor
        else:
            divisor = self.cache['batch_size'] * self.cache['seq_len']
            grad_logits /= float(divisor)

        return grad_logits
