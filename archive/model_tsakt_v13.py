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
        
        output = self._compute_tensor_attention(query, key, value, mask)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.total_size)
        
        return output, None

    def _compute_tensor_attention(self, query, key, value, mask=None):
        batch_size, num_heads, seq_length, head_size = query.shape
        
        q_tensor = self.tensor_proj_q(query)
        k_tensor = self.tensor_proj_k(key)
        
        q_tensor = q_tensor.transpose(1, 2)
        k_tensor = k_tensor.transpose(1, 2)
        
        q_tensor = q_tensor.unsqueeze(-1)
        k_tensor = k_tensor.unsqueeze(-2)
        
        interaction = torch.matmul(q_tensor, k_tensor)
        
        tensor_interaction = torch.einsum('bqij,ij->bqij', interaction, self.tensor_core)
        
        tensor_scores = tensor_interaction.sum(dim=(-1))
        
        if mask is not None:
            tensor_scores = tensor_scores.masked_fill(mask, -1e9)
        
        tensor_scores = tensor_scores.unsqueeze(1).expand(batch_size, num_heads, seq_length, seq_length)
        
        tensor_scores = tensor_scores / math.sqrt(self.tensor_rank)
        
        prob_attn = F.softmax(tensor_scores, dim=-1)
        
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        
        output = torch.matmul(prob_attn, value)
        
        return output


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
        attn_output, _ = self.tensor_attention(query, key, value, mask)
        
        attn_output = self.dropout(attn_output)
        output = self.linear_final(attn_output)
        
        return output, None


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
        
        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)
        
        if encode_pos:
            self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
            self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        
        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        
        self.attn_layers = nn.ModuleList([
            TransformerLayer(embed_size, num_heads, drop_prob, tensor_rank)
            for _ in range(num_attn_layers)
        ])
        
        self.lin_out = nn.Linear(embed_size, 1)

    def get_inputs(self, item_inputs, skill_inputs, label_inputs):
        item_inputs = self.item_embeds(item_inputs)
        skill_inputs = self.skill_embeds(skill_inputs)
        label_inputs = label_inputs.unsqueeze(-1).float()

        inputs = torch.cat([item_inputs, skill_inputs, item_inputs, skill_inputs], dim=-1)
        inputs[..., :self.embed_size] *= label_inputs
        inputs[..., self.embed_size:] *= 1 - label_inputs
        return inputs

    def get_query(self, item_ids, skill_ids):
        item_ids = self.item_embeds(item_ids)
        skill_ids = self.skill_embeds(skill_ids)
        query = torch.cat([item_ids, skill_ids], dim=-1)
        return query

    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        inputs = self.get_inputs(item_inputs, skill_inputs, label_inputs)
        query = self.get_query(item_ids, skill_ids)
        
        inputs = self.lin_in(inputs)
        query = self.lin_in(query)
        
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
