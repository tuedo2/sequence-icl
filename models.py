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
    Transformer block with self-attention and triangle masking
    """

    def __init__(self, n_embd=128, n_head=4):
        super(TransformerBlock, self).__init__()

        # Layer normalization for input
        self.norm1 = nn.LayerNorm(n_embd)

        # Multi-head self-attention
        self.multihead_attn = nn.MultiheadAttention(n_embd, num_heads=n_head, dropout=0.2, batch_first=True)

        # Layer normalization for attention output
        self.norm2 = nn.LayerNorm(n_embd)

        # Feed-forward neural network
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd)
        )

    def forward(self, x, padding_mask=None):
        b_size, layers, heads = x.shape
        mask = torch.triu(torch.ones(layers, layers, device=x.device), 1).bool()

        norm_x = self.norm1(x)
        x = self.multihead_attn(norm_x, norm_x, norm_x, attn_mask=mask, key_padding_mask=padding_mask)[0] + x
        
        norm_x = self.norm2(x)
        x = self.mlp(norm_x) + x

        return x
    
class TransformerModel(nn.Module):
    """
    Decoder-only Transformer with self-attention
    """

    def __init__(self, n_positions=128, n_embd=128, n_layer=4, n_head=4):
        super(TransformerModel, self).__init__()
        self.pos_emb = create_posistional_embedding(n_positions, n_embd)
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head) for _ in range(n_layer)
        ])

        self.out = nn.Linear(n_embd)