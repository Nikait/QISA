import torch
import torch.nn as nn
import torch.nn.functional as F

from QSAv2 import QSA


"""
Actual source:
https://github.com/karpathy/nanoGPT
"""

class TransformerBlock(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            n_embed: int, 
            block_size: int
        ):
        super(TransformerBlock, self).__init__()
        hidden_dim = n_embed // num_heads
        self.mhsa = MultiHeadSelfAttention(num_heads, hidden_dim, n_embed, block_size)
        self.feed_forward = FeedForward(n_embed)
        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mhsa(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class FeedForward(nn.Module):
    def __init__(
            self, 
            n_embed: int, 
            extend_width: int=4, 
            dropout: float=0.2
        ):
        super(FeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_embed, extend_width*n_embed), 
            nn.ReLU(),
            nn.Linear(extend_width*n_embed, n_embed), 
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self, 
            num_heads: int, 
            hidden_dim: int, 
            n_embed: int, 
            block_size: int, 
            dropout: float=0.2
        ):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.heads = nn.ModuleList([SingleHead(hidden_dim, n_embed, block_size) for _ in range(self.num_heads)])
        self.project = nn.Linear(n_embed, n_embed)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([sh(x) for sh in self.heads], dim=-1)
        out = self.project(out)
        out = self.drop(out)
        return out


class SingleHead(nn.Module):
    def __init__(
            self, 
            hidden_dim: int, 
            n_embed: int, 
            block_size: int, 
            dropout: float=0.2
        ):
        super(SingleHead, self).__init__()
        self.qsa = QSA(n_embed, block_size, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.qsa(x)
        return out


class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            block_size: int, 
            n_embed: int, 
            num_heads: int, 
            n_layers: int
        ):
        super(GPT, self).__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.positional_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(
            *[TransformerBlock(num_heads, n_embed, block_size) for _ in range(n_layers)],
        )
        self.norm = nn.LayerNorm(n_embed)        
        self.fc = nn.Linear(n_embed, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        token_embeddings = self.embedding(x) # B, T -> B, T, N_EMB
        positional_embedding = self.positional_embedding_table(torch.arange(T, device=x.device)) # T -> T, C
        token_embeddings = token_embeddings + positional_embedding # B, T, C + T, C -> B, T, C
        blocks_out = self.blocks(token_embeddings)
        blocks_out = self.norm(blocks_out)
        logits = self.fc(blocks_out) # B, T, N_EMB -> B, T, C
        logits = logits.reshape(B*T, self.vocab_size)
        return logits

    def generate(self, idx: torch.Tensor, max_tokens: int) -> torch.Tensor:
        t = idx.shape[1]
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            logits = logits.reshape(1, t, -1)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            if t < self.block_size:
                t += 1
        return idx
