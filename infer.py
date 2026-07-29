import os
import glob
import numpy as np
import pickle
import sys
from arr import as_device, print_device
from model import GPTDecoder

def main():
    print_device()

    try:
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("Loaded tokenizer successfully.")
    except FileNotFoundError:
        print("tokenizer.pkl not found! Please run train.py first.")
        sys.exit(1)

    vocab_size = tokenizer.vocab_size
    max_seq_len = 64
    embed_dim = 128
    num_layers = 4
    num_heads = 4

    print(f"Initializing model with vocab_size={vocab_size}, embed_dim={embed_dim}, num_layers={num_layers}, num_heads={num_heads}")
    model = GPTDecoder(vocab_size, embed_dim, num_layers, num_heads, max_seq_len=max_seq_len)

    weight_file = "maujmasti_weights_e10.npz"
    if not os.path.exists(weight_file):
        files = glob.glob("maujmasti_weights_e*.npz")
        if files:
            weight_file = sorted(files)[-1]
        else:
            print("No weight files found! Please run train.py first.")
            sys.exit(1)
    print(f"Loading weights from: {weight_file}")

    model.load_weights(weight_file)
    print("Loaded weights successfully.")

    print(f"\n--- Model ready! Type 'quit' to exit ---")
    while True:
        prompt = input("\nEnter starting text: ")
        if prompt.lower() in ('quit', 'exit'):
            break

        try:
            tokens = tokenizer.encode(prompt)
            if not tokens:
                print("Empty prompt. Try again.")
                continue

            temp = 0.7
            gen_len = 80
            if len(prompt.split()) > 1:
                parts = prompt.rsplit(maxsplit=1)
                try:
                    temp = float(parts[1])
                    prompt = parts[0]
                    tokens = tokenizer.encode(prompt)
                except ValueError:
                    pass

            start_tokens = as_device(np.array([tokens], dtype=np.int32))

            generated = model.generate(start_tokens, max_length=gen_len, temperature=temp)
            output_text = tokenizer.decode(generated[0])
            print(f"\nOutput:\n{output_text}")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
