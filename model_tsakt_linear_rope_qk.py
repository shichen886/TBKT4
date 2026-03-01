import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Rotary Position Embedding (RoPE) for Q/K
# ------------------------------
class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) applied to Q/K in attention
    This is the correct usage: apply RoPE to Q/K, not to embedding
    """

    def __init__(self, head_dim, max_len=500):
        super().__init__()
        self.head_dim = head_dim
        self.dim = head_dim // 2  # Split into two halves for sin/cos
        self.max_len = max_len

        # Precompute rotation angles for all positions
        self.register_buffer(
            'freqs',
            self._precompute_freqs(max_len)
        )

    def _precompute_freqs(self, max_len):
        """
        Precompute rotation angles for all positions
        Returns: [max_len, dim]
        """
        position = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.arange(self.dim, dtype=torch.float32)
        
        # Create frequency scaling (theta_i = 10000^(-2i/d))
        theta = 1.0 / (10000 ** (freqs / self.dim))
        
        # Compute angles: position * theta
        angles = position.unsqueeze(-1) * theta.unsqueeze(0)  # [max_len, dim]
        
        return angles

    def rotate(self, x, position_ids):
        """
        Apply rotation to Q or K
        
        Args:
            x: [B, H, L, Dh] - Q or K tensor
            position_ids: [B, L] - position indices
        """
        batch_size, num_heads, seq_len, head_dim = x.shape

        # Get rotation angles for current positions
        angles = self.freqs[position_ids]  # [B, L, dim]
        angles = angles.unsqueeze(1)  # [B, 1, L, dim]

        # Compute sin and cos
        sin = torch.sin(angles)
        cos = torch.cos(angles)

        # Split x into two halves
        x1 = x[..., :self.dim]  # [B, H, L, dim]
        x2 = x[..., self.dim:]  # [B, H, L, dim]

        # Apply rotation
        x_rotated = torch.cat([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1)

        return x_rotated


# ------------------------------
# Linear (O(L·R)) Tensor Attention with RoPE on Q/K
# ------------------------------
class LinearTensorSelfAttention_RoPE(nn.Module):
    """
    True linear‑complexity attention with RoPE on Q/K:
        Output = φ(RoPE(Q)) · (RoPE(φ(K))^T · V)

    Complexity: O(L · R · D)
    No L×L matrix is ever constructed.
    
    Key difference: RoPE is applied to Q/K in attention, not to embedding
    """

    def __init__(self, embed_size, num_heads, drop_prob=0.1, tensor_rank=32, max_len=500):
        super().__init__()

        assert embed_size % num_heads == 0

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.tensor_rank = tensor_rank
        self.max_len = max_len

        # QKV projections
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

        # Low‑rank feature maps φ(·)
        self.q_feature = nn.Linear(self.head_dim, tensor_rank)
        self.k_feature = nn.Linear(self.head_dim, tensor_rank)

        # RoPE for Q/K
        self.rope = RotaryPositionEmbedding(self.head_dim, max_len)

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

        # Apply RoPE to Q/K BEFORE feature mapping
        position_ids = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        Q = self.rope.rotate(Q, position_ids)  # [B,H,L,Dh]
        K = self.rope.rotate(K, position_ids)  # [B,H,L,Dh]

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
class LinearTransformerLayer_RoPE(nn.Module):
    """
    One linear‑complexity transformer layer with RoPE on Q/K
    """

    def __init__(self, embed_size, num_heads, drop_prob=0.1, tensor_rank=32, max_len=500):
        super().__init__()
        self.attention = LinearTensorSelfAttention_RoPE(embed_size, num_heads, drop_prob, tensor_rank, max_len)
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
# TSAKT‑Linear with RoPE on Q/K (Correct Usage)
# ------------------------------
class TSAKT_Linear_RoPE_QK(nn.Module):
    """
    TSAKT‑Linear with Rotary Position Embedding on Q/K (Correct Usage)
    
    Key difference from previous implementation:
    - Previous: RoPE applied to embedding (WRONG)
    - This: RoPE applied to Q/K in attention (CORRECT)
    
    This is the standard usage in KT papers:
    - embedding + pos → worse performance
    - attention with pos (on Q/K) → better performance
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

        self.input_proj = nn.Linear(embed_size * 2, embed_size)

        # Transformer layers with RoPE on Q/K
        self.layers = nn.ModuleList(
            [
                LinearTransformerLayer_RoPE(embed_size, num_heads, drop_prob, tensor_rank, max_len)
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
        x = torch.cat([item, skill], dim=-1)
        x = self.input_proj(x)

        # transformer layers (RoPE is applied inside attention to Q/K)
        for layer in self.layers:
            x = layer(x, mask)

        # output projection
        out = self.out(x)

        return out

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item = self.item_embed(item_inputs)
        skill = self.skill_embed(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()
        return item, skill, label_inputs
