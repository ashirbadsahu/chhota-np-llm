from arr import xp

class LMTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.lr = learning_rate

    def train_step(self, batch_tokens):
        input_ids = batch_tokens[:, :-1]
        target_ids = batch_tokens[:, 1:]

        return 0.0

    def train(self, train_data, num_epochs, batch_size):
        pass


import math

class WarmupCosineScheduler:
    def __init__(self, max_lr, min_lr, total_steps, warmup_steps):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step / self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
