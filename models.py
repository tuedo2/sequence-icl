import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import math

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_posistional_embedding(max_len, n_embd):
    """
    Args:
        max_len: maximum length supported for positional embeddings
        n_embd: embedding dimension
    Returns:
        pe: [max_len, n_embd] positional embedding matrix
    """
    pe = torch.zeros(max_len, n_embd, dtype=float)

    for i in range(n_embd // 2):
        denom = np.exp(2 * i / n_embd * np.log(10000))
        for pos in range(max_len):
            pe[pos, 2 * i] = np.sin(pos / denom)
            pe[pos, 2 * i + 1] = np.cos(pos / denom)
    
    return pe

class TransformerBlock(nn.Module):
    """
    Implementation inspired from https://web.stanford.edu/~jurafsky/slp3/9.pdf
    """
    def __init__(self, n_embd=512, n_heads=4, dropout=0.1):
        self.norm1 = nn.LayerNorm(n_embd)
        self.multiheadattention = nn.MultiheadAttention(n_embd, n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        t1 = self.norm1(x)
        t2 = self.multiheadattention(x, x, x, causal_mask = True)
        t3 = t1 + t2
        t4 = self.norm2(t3)
        t5 = self.feedforward(t4)
        h = t5 + t3

        return h