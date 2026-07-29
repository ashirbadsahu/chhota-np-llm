import numpy as np
import time
import pickle
import os
from arr import xp, as_device, print_device
from tokenizer import BBPETokenizer
from data import LanguageModelDataset, DataLoader
from model import GPTDecoder
from loss import CrossEntropyLoss
from optim import AdamOptimizer
from utils import WarmupCosineScheduler

def count_parameters(model):
    total_params = 0
    for name, param in model.get_params():
        total_params += param.size
    return total_params

def get_grads_list(model, grads_dict):
    grads_list = []
    for name, param in model.get_params():
        grads_list.append((name, grads_dict[name]))
    return grads_list

def to_float(val):
    if hasattr(val, 'get'):
        return float(val.get())
    return float(val)

def main():
    print_device()

    print("\nLoading data...")
    with open('tiny-shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    texts = text.split('\n\n')

    print("Training tokenizer...")
    vocab_size = 1024
    tokenizer = BBPETokenizer(vocab_size=vocab_size)
    tokenizer.train(texts, verbose=False)
    print(f"Tokenizer trained with vocab size {tokenizer.vocab_length()}")

    max_seq_len = 64
    batch_size = 8

    dataset = LanguageModelDataset(texts, tokenizer, max_seq_len=max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset created with {len(dataset)} sequences.")

    embed_dim = 128
    num_layers = 4
    num_heads = 4

    print(f"Initializing GPTDecoder(vocab_size={vocab_size}, embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads})")
    model = GPTDecoder(vocab_size, embed_dim, num_layers, num_heads, max_seq_len=max_seq_len)
    print(f"Model parameters: {count_parameters(model):,}")

    loss_fn = CrossEntropyLoss()
    optimizer = AdamOptimizer(model.get_params(), lr=1.5e-3, max_grad_norm=1.0)

    num_epochs = 10
    steps_per_epoch = len(dataloader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    scheduler = WarmupCosineScheduler(max_lr=1.5e-3, min_lr=1e-5, total_steps=total_steps, warmup_steps=warmup_steps)

    print(f"\nTraining: {num_epochs} epochs, {total_steps} steps, warmup={warmup_steps}")
    print(f"LR schedule: warmup to 1.5e-3, cosine decay to 1e-5\n")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_steps = 0

        for step, (batch, mask) in enumerate(dataloader):
            input_ids = batch[:, :-1]
            target_ids = batch[:, 1:]
            batch_mask = mask[:, 1:]

            logits = model.forward(input_ids)
            loss = loss_fn.forward(logits, target_ids, mask=batch_mask)

            grad_logits = loss_fn.backward()
            grads_dict = model.backward(grad_logits)

            global_step = step + epoch * steps_per_epoch
            lr = scheduler.get_lr(global_step)
            optimizer.lr = lr

            grads_list = get_grads_list(model, grads_dict)
            optimizer.step(model.get_params(), grads_list)

            epoch_loss += to_float(loss)
            n_steps += 1

            if (step + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Epoch {epoch + 1}/{num_epochs} | Step {step + 1}/{steps_per_epoch} | Loss: {to_float(loss):.4f} | LR: {to_float(lr):.6f} | Time: {elapsed:.1f}s")

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch + 1}/{num_epochs} completed | Avg Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")

        model.save_weights(f"chhota_np_llm_weights_e{epoch + 1}.npz")
        print(f"  Weights saved: chhota_np_llm_weights_e{epoch + 1}.npz")

    print(f"\nTraining finished in {time.time() - start_time:.1f}s!")

    model.load_weights(f"chhota_np_llm_weights_e{num_epochs}.npz")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Tokenizer saved: tokenizer.pkl")

    print("\n--- Generated samples ---")
    prompts = ["Please you", "Once upon", "To be or not", "The sun"]
    for prompt in prompts:
        start_token = np.array([tokenizer.encode(prompt)])
        generated = model.generate(start_token, max_length=80, temperature=0.7)
        decoded = tokenizer.decode(generated[0])
        print(f"Prompt: \"{prompt}\"")
        print(f"Output: {decoded}\n")

if __name__ == "__main__":
    main()
