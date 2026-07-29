import numpy as np

class Evaluator:
    def __init__(self, model, loss_fn):
        self.model = model
        self.loss_fn = loss_fn

    def evaluate(self, val_loader):
        """
        Compute validation metrics
        """
        total_loss = 0.0
        num_batches = len(val_loader)

        if num_batches == 0:
            return {'loss': float('inf'), 'perplexity': float('inf')}

        for batch in val_loader:
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]

            # Forward pass
            logits = self.model.forward(input_ids)
            loss = self.loss_fn.forward(logits, target_ids)

            total_loss += loss
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)

        return {'loss': avg_loss, 'perplexity': perplexity}