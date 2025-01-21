import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass

import math

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
    B: batch size
    L: context length
    H: embedding dimension
"""

""" 
    Much code and styling derived from https://github.com/karpathy/nanoGPT/tree/master
    Knowledge derived from https://web.stanford.edu/~jurafsky/slp3/9.pdf
"""

class Head(nn.Module):
    """ One head of self-attention with future context masking """

    def __init__(self, n_embd, head_size, context_length=64, dropout=0.0):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, H = x.shape
        k = self.key(x)     # (B, L, H)
        q = self.query(x)   # (B, L, H)
        # compute attention scores
        wei = q @ k.transpose(-2, -1) * (H ** -0.5)     # (B, L, H) @ (B, H, L) -> (B, L, L)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, L, L)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v       # (B, L, L) @ (B, L, H) -> (B, L, H)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, n_embd, num_heads, head_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class CausalSelfAttention(nn.Module):
    """ faster version of MultiHeadAttention derived from nanoGPT by parallelizing over the heads in forward pass """

    def __init__(self, n_embd, n_head, dropout=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.resid_dropout = nn.Dropout(dropout)

        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention is faster but support is only in PyTorch >= 2.0
        self.flash = hasattr(F, 'scaled_dot_product_attention')

    def forward(self, x):
        B, L, H = x.shape # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, H, self.n_head, H // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, H, self.n_head, H // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, H, self.n_head, H // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropoutp=(self.dropout if self.training else 0.0), is_causal=True)
        
        y = y.transpose(1, 2).contiguous().view(B, L, H) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class TransformerBlock(nn.Module):

    def __init__(self, n_embd=512, num_heads=4, dropout=0.0):
        head_size = n_embd // num_heads
        self.norm1 = nn.LayerNorm(n_embd)
        self.multihead_attention = CausalSelfAttention(n_embd, num_heads, head_size, dropout=dropout)
        self.norm2 = nn.LayerNorm(n_embd)
        self.feedforward = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        t1 = self.norm1(x)
        t2 = self.multihead_attention(x)
        t3 = t1 + t2
        t4 = self.norm2(t3)
        t5 = self.feedforward(t4)
        h = t5 + t3

        return h

@dataclass
class Config:
    context_length: int = 64 
    vocab_size: int = 256 # GPT-2 vocab_size of 50257
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 512
    dropout: float = 0.0

class TransformerModel(nn.Module):
    """ base transformer model with token and positional embedding """

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.context_length, config.n_embd)
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config.n_embd, config.n_head, config.dropout) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.n_embd)      # added final layer norm
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)    # Unembedding
    
    def forward(self, idx, targets=None):
        B, L = idx.shape

        # idx and targets are both (B, L) tensor of integers representing tokens
        tok_embd = self.token_embedding(idx)    # (B, L, H)
        pos_embd = self.position_embedding(torch.arange(L, device=DEVICE))  # (L, H)
        x = tok_embd + pos_embd     # (B, L, H)
        x = self.transformer_blocks(x)  # (B, L, H)
        x = self.norm_f(x)          # (B, L, H)
        logits = self.lm_head(x)    # (B, L, vocab_size)

        if targets is None:
            loss = None
        else:
            B, L, V = logits.shape
            logits = logits.view(B * L, V)
            targets = targets.view(B * L)
            loss = F.cross_entropy(logits, targets)     # In regression case, loss is not defined like this
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Be in model.eval() mode.
        """

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop at context_length
            idx_cond = idx if idx.size(1) <= self.config.context_length else idx[:, -self.config.context_length:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
        

