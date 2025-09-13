import torch
from torch import nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, mask=None): #(batch_size, seq_len, embed_dim)
        batch_size, seq_len, embed_dim = x.shape
        Q = self.q_proj(x) # (batch_size, seq_len, embed_dim)
        K = self.k_proj(x)
        V = self.v_proj(x)

        scores = torch.matmul(Q,K.transpose(1,2))
        scale = math.sqrt(K.size(-1))
        scores = scores/scale
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)

        attention = torch.matmul(V, weights)
        return attention, weights