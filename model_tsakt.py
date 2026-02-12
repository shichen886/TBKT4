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
        
        self.linear_global = nn.Linear(embed_size, num_heads)
        
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
        
        score_glo = self.linear_global(key).unsqueeze(-1).transpose(1, 2)
        
        query = query.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1)) + score_glo
        
        tensor_scores = self._compute_tensor_attention(query, key)
        # Expand tensor_scores to match the shape of scores
        tensor_scores = tensor_scores.unsqueeze(-1).expand_as(scores)
        scores = scores + tensor_scores
        
        scores = scores / math.sqrt(query.size(-1))
        
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
        
        tensor_scores = tensor_interaction.sum(dim=(-1, -2))
        
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

    def forward(self, query, key, value, mask=None):
        out, self.prob_attn = self.tensor_attention(query, key, value, mask)
        return out


class TSAKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads,
                 encode_pos, max_pos, drop_prob, tensor_rank=3):
        """Tensor Self-Attentive Knowledge Tracing.

        Arguments:
            num_items (int): number of items
            num_skills (int): number of skills
            embed_size (int): input embedding and attention dot-product dimension
            num_attn_layers (int): number of attention layers
            num_heads (int): number of parallel attention heads
            encode_pos (bool): if True, use relative position embeddings
            max_pos (int): number of position embeddings to use
            drop_prob (float): dropout probability
            tensor_rank (int): rank of tensor core for tensor attention
        """
        super(TSAKT, self).__init__()
        self.embed_size = embed_size
        self.encode_pos = encode_pos
        self.tensor_rank = tensor_rank

        self.item_embeds = nn.Embedding(num_items + 1, embed_size // 2, padding_idx=0)
        self.skill_embeds = nn.Embedding(num_skills + 1, embed_size // 2, padding_idx=0)

        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)

        self.lin_in = nn.Linear(2 * embed_size, embed_size)
        self.attn_layers = clone(TensorMultiHeadAttention(embed_size, num_heads, drop_prob, tensor_rank), 
                                  num_attn_layers)
        self.dropout = nn.Dropout(p=drop_prob)
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
        inputs = F.relu(self.lin_in(inputs))

        query = self.get_query(item_ids, skill_ids)

        mask = future_mask(inputs.size(-2))
        if inputs.is_cuda:
            mask = mask.cuda()

        outputs = self.dropout(self.attn_layers[0](query, inputs, inputs, mask))
        for l in self.attn_layers[1:]:
            residual = l(query, outputs, outputs, mask)
            outputs = self.dropout(outputs + F.relu(residual))

        return self.lin_out(outputs)
