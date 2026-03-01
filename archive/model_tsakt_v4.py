import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


def clone(module, num):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(num)])


class TensorSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob, tensor_rank=3):
        super(TensorSelfAttention, self).__init__()
        assert embed_size % num_heads == 0
        self.total_size = embed_size
        self.head_size = embed_size // num_heads
        self.num_heads = num_heads
        self.tensor_rank = tensor_rank
        
        self.linear_q = nn.Linear(embed_size, embed_size)
        self.linear_k = nn.Linear(embed_size, embed_size)
        self.linear_v = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.tensor_proj_q = nn.Linear(self.head_size, tensor_rank)
        self.tensor_proj_k = nn.Linear(self.head_size, tensor_rank)
        
        self.tensor_core = nn.Parameter(torch.randn(tensor_rank, tensor_rank))
        nn.init.xavier_uniform_(self.tensor_core)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_length = query.shape[:2]
        
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        query = self.linear_q(query)
        key = self.linear_k(key)
        value = self.linear_v(value)
        
        query = query.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = self._compute_tensor_attention(query, key)
        
        scores = scores / math.sqrt(self.tensor_rank)
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        
        prob_attn = F.softmax(scores, dim=-1)
        
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        
        output = torch.matmul(prob_attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        
        return output, prob_attn

    def _compute_tensor_attention(self, query, key):
        batch_size, num_heads, seq_length, head_size = query.shape
        
        q_tensor = self.tensor_proj_q(query)
        k_tensor = self.tensor_proj_k(key)
        
        q_tensor = q_tensor.unsqueeze(-1)
        k_tensor = k_tensor.unsqueeze(-2)
        
        interaction = torch.matmul(q_tensor, k_tensor)
        
        tensor_interaction = torch.einsum('bhqij,ij->bhqij', interaction, self.tensor_core)
        
        tensor_scores = tensor_interaction.sum(dim=(-1))
        
        return tensor_scores


class TensorMultiHeadAttention(nn.Module):
    def __init__(self, total_size, num_heads, drop_prob, tensor_rank=3):
        super(TensorMultiHeadAttention, self).__init__()
        assert total_size % num_heads == 0
        self.total_size = total_size
        self.head_size = total_size // num_heads
        self.num_heads = num_heads
        self.tensor_rank = tensor_rank
        
        self.tensor_attention = TensorSelfAttention(total_size, num_heads, drop_prob, tensor_rank)
        self.dropout = nn.Dropout(p=drop_prob)
        
        self.linear_final = nn.Linear(total_size, total_size)

    def forward(self, query, key, value, mask=None):
        attn_output, attn_weights = self.tensor_attention(query, key, value, mask)
        
        attn_output = self.dropout(attn_output)
        output = self.linear_final(attn_output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    def __init__(self, embed_size, drop_prob=0.2):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(embed_size, embed_size)
        self.linear_2 = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(p=drop_prob)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob, tensor_rank=3):
        super(TransformerLayer, self).__init__()
        self.self_attn = TensorMultiHeadAttention(embed_size, num_heads, drop_prob, tensor_rank)
        self.feed_forward = FeedForward(embed_size, drop_prob)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, mask=None):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TSAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob, tensor_rank=3):
        super(TSAKT, self).__init__()
        self.num_items = num_items
        self.num_skills = num_skills
        self.embed_size = embed_size
        self.encode_pos = encode_pos
        self.max_pos = max_pos
        self.tensor_rank = tensor_rank
        
        self.item_embeds = nn.Embedding(num_items + 1, embed_size)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size)
        
        if encode_pos:
            self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
            self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        
        self.lin_in = nn.Linear(embed_size * 2, embed_size)
        
        self.attn_layers = nn.ModuleList([
            TransformerLayer(embed_size, num_heads, drop_prob, tensor_rank)
            for _ in range(num_attn_layers)
        ])
        
        self.lin_out = nn.Linear(embed_size, 1)

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        item_embeds = self.item_embeds(item_inputs)
        skill_embeds = self.skill_embeds(skill_inputs)
        
        inputs = torch.cat([item_embeds, skill_embeds], dim=-1)
        inputs = self.lin_in(inputs)
        
        if self.encode_pos:
            seq_length = inputs.shape[1]
            positions = torch.arange(seq_length, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
            pos_key_embeds = self.pos_key_embeds(positions)
            pos_value_embeds = self.pos_value_embeds(positions)
        else:
            pos_key_embeds = None
            pos_value_embeds = None
        
        for layer in self.attn_layers:
            inputs = layer(inputs)
        
        outputs = self.lin_out(inputs)
        return outputs

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_embeds = self.item_embeds(item_inputs)
        skill_embeds = self.skill_embeds(skill_inputs)
        
        inputs = torch.cat([item_embeds, skill_embeds], dim=-1)
        inputs = self.lin_in(inputs)
        
        if self.encode_pos:
            seq_length = inputs.shape[1]
            positions = torch.arange(seq_length, device=inputs.device).unsqueeze(0).expand(inputs.shape[0], -1)
            pos_key_embeds = self.pos_key_embeds(positions)
            pos_value_embeds = self.pos_value_embeds(positions)
        else:
            pos_key_embeds = None
            pos_value_embeds = None
        
        return inputs
