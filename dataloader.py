import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T, address):
        self.B = B
        self.T = T

        with open(address, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        self.current_batch = 0
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch ={len(self.tokens)//(B*T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_batch : (self.current_batch + B * T + 1)]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        if (self.current_batch + B * T) > len(self.tokens):
            self.current_batch = 0

        return x, y
