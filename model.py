"""A small decoder-only GPT that learns to predict the next token."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass(frozen=True)
class GPTConfig:
    vocab_size: int                       # Number of token IDs in the tokenizer.
    block_size: int = 1024                # Context length, not total generation length.
    n_layer: int = 6                      # Number of stacked Transformer blocks.
    n_head: int = 4                       # Each head uses a slice of the token vector.
    n_embd: int = 256                     # Width of each token's vector.
    dropout: float = 0.1                  # Fraction of activations dropped during training.


class CausalSelfAttention(nn.Module):
    """Let each token gather information from itself and earlier tokens."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        # One projection produces query, key and value vectors of equal width.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head

        # A position may attend to itself and earlier positions, never the future.
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, width = x.shape
        head_size = width // self.n_head
        q, k, v = self.c_attn(x).split(width, dim=2)
        # Give each head its own vectors: [batch, heads, tokens, head_size].
        q = q.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, head_size).transpose(1, 2)

        # Compare queries with keys; scaling keeps the scores from growing too large.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_size)
        # Future positions receive zero probability after softmax.
        scores = scores.masked_fill(~self.mask[:, :, :seq_len, :seq_len], float("-inf"))
        weights = self.attn_dropout(F.softmax(scores, dim=-1))
        # Take a weighted sum of values, then join the heads back into one vector.
        x = weights @ v
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, width)
        # Learn how to mix information from the joined heads.
        return self.resid_dropout(self.c_proj(x))


class MLP(nn.Module):
    """Transform each token independently; attention handles mixing between tokens."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer block: attention, then an MLP, with residual additions."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize before each sublayer, residual connections after each sublayer.
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class GPT(nn.Module):
    """Turn token IDs into context-aware vectors and scores for the next token."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config 
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # Token identity.
            wpe=nn.Embedding(config.block_size, config.n_embd),  # Position in the window.
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Share the input embedding and output weights to reduce parameter count.
        self.lm_head.weight = self.transformer.wte.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError("input is longer than block_size")

        positions = torch.arange(seq_len, device=idx.device)
        # Each position starts with both token information and its position vector.
        x = self.transformer.wte(idx) + self.transformer.wpe(positions)
        x = self.transformer.drop(x)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is None:
            # Generation needs only the last position's next-token prediction.
            return self.lm_head(x[:, -1:, :]), None
        logits = self.lm_head(x)
        # Treat every batch position as one classification over the vocabulary.
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Append sampled tokens and return the prompt together with its continuation."""
        if idx.size(1) == 0 or temperature <= 0:
            raise ValueError("prompt must be nonempty and temperature must be positive")
        for _ in range(max_new_tokens):
            # The output may grow longer, but each prediction sees only this window.
            context = idx[:, -self.config.block_size:]
            logits, _ = self(context)
            # A lower temperature makes high-scoring tokens more likely.
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                # Exclude tokens outside the highest-scoring candidates.
                threshold = torch.topk(logits, min(top_k, logits.size(-1))).values[:, -1:]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            # Sample from the probabilities rather than always choosing the top token.
            next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
