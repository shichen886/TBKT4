import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Linear (O(L·R)) Tensor Attention
# ------------------------------
class LinearTensorSelfAttention(nn.Module):
    """
    True linear‑complexity attention:
        Output = φ(Q) · (φ(K)^T · V)

    Complexity: O(L · R · D)
    No L×L matrix is ever constructed.
    """

    def __init__(self, embed_size, num_heads, drop_prob=0.1, tensor_rank=32):
        super().__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.tensor_rank = tensor_rank

        # QKV projections
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

        # Low‑rank feature maps φ(·)
        self.q_feature = nn.Linear(self.head_dim, tensor_rank)
        self.k_feature = nn.Linear(self.head_dim, tensor_rank)

        self.dropout = nn.Dropout(drop_prob)

        self.out_proj = nn.Linear(embed_size, embed_size)

    # ------------------------------
    # positive kernel feature map
    # ------------------------------
    def phi(self, x):
        # ELU+1 ensures positivity → stable normalization
        return F.elu(x) + 1

    # ------------------------------
    # forward
    # ------------------------------
    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L] (optional causal handled outside if needed)
        """

        B, L, _ = x.shape

        # ---- QKV ----
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ---- reshape to heads ----
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,Dh]
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # ---- low‑rank feature projection ----
        q = self.phi(self.q_feature(q))  # [B,H,L,R]
        k = self.phi(self.k_feature(k))  # [B,H,L,R]

        # ---- optional mask on K/V ----
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]
            k = k * mask
            v = v * mask

        # ======================================================
        #   TRUE LINEAR ATTENTION CORE
        # ======================================================

        # Step1: aggregate KV  → [B,H,R,Dh]
        kv = torch.einsum("bhlr,bhld->bhrd", k, v)

        # Step2: compute normalization term
        z = 1.0 / (torch.einsum("bhlr,bhr->bhl", q, k.sum(dim=2)) + 1e-6)

        # Step3: final output  → [B,H,L,Dh]
        out = torch.einsum("bhlr,bhrd,bhl->bhld", q, kv, z)

        # ---- merge heads ----
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_size)

        return self.out_proj(self.dropout(out))


# ------------------------------
# FeedForward
# ------------------------------
class FeedForward(nn.Module):
    def __init__(self, embed_size, drop_prob=0.1, expansion=4):
        super().__init__()
        hidden = embed_size * expansion
        self.net = nn.Sequential(
            nn.Linear(embed_size, hidden),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden, embed_size),
            nn.Dropout(drop_prob),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------
# Transformer Block
# ------------------------------
class LinearTransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob=0.1, tensor_rank=32):
        super().__init__()

        self.attn = LinearTensorSelfAttention(embed_size, num_heads, drop_prob, tensor_rank)
        self.ffn = FeedForward(embed_size, drop_prob)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ------------------------------
# TSAKT‑Linear Model WITHOUT Positional Encoding
# ------------------------------
class TSAKT_Linear_NoPos(nn.Module):
    """
    Linear‑complexity TSAKT WITHOUT Positional Encoding
    Total attention cost: O(L · R · D)
    """

    def __init__(
        self,
        num_items,
        num_skills,
        embed_size=128,
        num_layers=2,
        num_heads=4,
        tensor_rank=32,
        max_len=500,
        drop_prob=0.1,
    ):
        super().__init__()

        self.item_embed = nn.Embedding(num_items + 1, embed_size)
        self.skill_embed = nn.Embedding(num_skills + 1, embed_size)

        # NO positional encoding - removed for ablation study

        self.input_proj = nn.Linear(embed_size * 2, embed_size)

        self.layers = nn.ModuleList(
            [
                LinearTransformerLayer(embed_size, num_heads, drop_prob, tensor_rank)
                for _ in range(num_layers)
            ]
        )

        self.out = nn.Linear(embed_size, 1)

    # ------------------------------
    # forward
    # ------------------------------
    def forward(self, item_ids, skill_ids, mask=None):
        """
        item_ids, skill_ids: [B, L]
        mask: [B, L]  (1 = valid, 0 = pad)
        """

        B, L = item_ids.shape

        item = self.item_embed(item_ids)
        skill = self.skill_embed(skill_ids)

        x = torch.cat([item, skill], dim=-1)
        x = self.input_proj(x)

        # NO positional encoding - removed for ablation study

        # transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # output projection
        out = self.out(x)

        return out