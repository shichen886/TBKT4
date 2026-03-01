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
        mask: [B, L] (1 = valid, 0 = pad)
        """
        B, L, D = x.shape

        # QKV
        Q = self.q_proj(x)  # [B, L, D]
        K = self.k_proj(x)  # [B, L, D]
        V = self.v_proj(x)  # [B, L, D]

        # Multi-head
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B,H,L,Dh]
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Low‑rank feature maps
        Q = self.phi(self.q_feature(Q))  # [B,H,L,R]
        K = self.phi(self.k_feature(K))  # [B,H,L,R]

        # Optional mask on K/V
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # [B,1,L,1]
            K = K * mask
            V = V * mask

        # ======================================================
        #   TRUE LINEAR ATTENTION CORE
        # ======================================================

        # Step1: aggregate KV  → [B,H,R,Dh]
        KV = torch.einsum("bhlr,bhld->bhrd", K, V)

        # Step2: compute normalization term
        Z = 1.0 / (torch.einsum("bhlr,bhr->bhl", Q, K.sum(dim=2)) + 1e-6)

        # Step3: final output  → [B,H,L,Dh]
        out = torch.einsum("bhlr,bhrd,bhl->bhld", Q, KV, Z)

        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(self.dropout(out))

        return out


# ------------------------------
# Linear Transformer Layer
# ------------------------------
class LinearTransformerLayer(nn.Module):
    """
    One linear‑complexity transformer layer
    """

    def __init__(self, embed_size, num_heads, drop_prob=0.1, tensor_rank=32):
        super().__init__()
        self.attention = LinearTensorSelfAttention(embed_size, num_heads, drop_prob, tensor_rank)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Dropout(drop_prob),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        """
        x: [B, L, D]
        mask: [B, L] (1 = valid, 0 = pad)
        """
        # Self-attention with Pre-LN
        x = x + self.dropout(self.attention(self.norm1(x), mask))

        # FFN with Pre-LN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# ------------------------------
# Gate Fusion Mechanism for Position Encoding
# ------------------------------
class PositionGate(nn.Module):
    """
    Gate fusion mechanism for position encoding
    Dynamically controls the weight of position encoding
    """

    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.GELU(),
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid()
        )

    def forward(self, content, position):
        """
        Args:
            content: [B, L, D] - content embeddings (item + skill)
            position: [B, L, D] - position embeddings
        Returns:
            fused: [B, L, D] - gated fusion
        """
        # Compute gate weight
        combined = torch.cat([content, position], dim=-1)
        gate_weight = self.gate(combined)  # [B, L, D]

        # Apply gate
        fused = gate_weight * position + (1 - gate_weight) * content

        return fused, gate_weight


# ------------------------------
# TSAKT‑Linear with Gate Fusion
# ------------------------------
class TSAKT_Linear_Gate(nn.Module):
    """
    TSAKT‑Linear with Gate Fusion Mechanism
    Uses gate to dynamically control position encoding weight
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

        # Simple position embedding
        self.pos_embed = nn.Embedding(max_len, embed_size)

        # Gate fusion mechanism
        self.pos_gate = PositionGate(embed_size)

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
        mask: [B, L] (1 = valid, 0 = pad)
        """
        B, L = item_ids.shape

        item = self.item_embed(item_ids)
        skill = self.skill_embed(skill_ids)
        content = torch.cat([item, skill], dim=-1)
        content = self.input_proj(content)

        # Position embedding
        pos = torch.arange(L, device=content.device).unsqueeze(0)  # [1, L]
        position = self.pos_embed(pos)  # [1, L, D]
        position = position.expand(B, -1, -1)  # [B, L, D]

        # Gate fusion
        fused, gate_weight = self.pos_gate(content, position)

        # transformer layers
        for layer in self.layers:
            fused = layer(fused, mask)

        # output projection
        out = self.out(fused)

        return out

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item = self.item_embed(item_inputs)
        skill = self.skill_embed(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()
        return item, skill, label_inputs