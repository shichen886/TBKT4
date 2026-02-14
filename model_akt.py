import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AKT(nn.Module):
    def __init__(self, num_items, num_skills, embed_size, num_attn_layers, num_heads, drop_prob, max_pos):
        """
        Attentive Knowledge Tracing (AKT) Model
        
        Args:
            num_items: number of items
            num_skills: number of skills
            embed_size: dimension of embedding
            num_attn_layers: number of attention layers
            num_heads: number of attention heads
            drop_prob: dropout probability
            max_pos: maximum position for positional encoding
        """
        super(AKT, self).__init__()
        
        self.num_items = num_items
        self.num_skills = num_skills
        self.embed_size = embed_size
        self.num_attn_layers = num_attn_layers
        self.num_heads = num_heads
        self.max_pos = max_pos
        
        self.item_embeds = nn.Embedding(num_items, embed_size)
        self.skill_embeds = nn.Embedding(num_skills, embed_size)
        
        self.pos_key_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        self.pos_value_embeds = nn.Embedding(max_pos, embed_size // num_heads)
        
        self.lin_in = nn.Linear(embed_size * 2, embed_size)
        
        self.attn_layers = nn.ModuleList([
            AKTAttentionLayer(embed_size, num_heads, drop_prob, max_pos)
            for _ in range(num_attn_layers)
        ])
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.lin_out = nn.Linear(embed_size, 1)
        
    def forward(self, item_inputs, skill_inputs, label_inputs, item_ids, skill_ids):
        """
        Forward pass
        
        Args:
            item_inputs: item input sequence [batch_size, seq_len]
            skill_inputs: skill input sequence [batch_size, seq_len]
            label_inputs: label input sequence [batch_size, seq_len]
            item_ids: item ids [batch_size, seq_len]
            skill_ids: skill ids [batch_size, seq_len]
        
        Returns:
            predictions [batch_size, seq_len]
        """
        batch_size, seq_len = item_inputs.shape
        
        item_embeds = self.item_embeds(item_inputs)
        skill_embeds = self.skill_embeds(skill_inputs)
        
        x = torch.cat([item_embeds, skill_embeds], dim=-1)
        x = self.lin_in(x)
        
        for attn_layer in self.attn_layers:
            x, _ = attn_layer(x, item_ids, skill_ids)
        
        x = self.dropout(x)
        output = self.lin_out(x)
        
        return output.squeeze(-1)


class AKTAttentionLayer(nn.Module):
    def __init__(self, embed_size, num_heads, drop_prob, max_pos):
        """
        AKT Attention Layer
        
        Args:
            embed_size: dimension of embedding
            num_heads: number of attention heads
            drop_prob: dropout probability
            max_pos: maximum position for positional encoding
        """
        super(AKTAttentionLayer, self).__init__()
        
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.max_pos = max_pos
        
        self.linear_q = nn.Linear(embed_size, embed_size)
        self.linear_k = nn.Linear(embed_size, embed_size)
        self.linear_v = nn.Linear(embed_size, embed_size)
        
        self.dropout = nn.Dropout(drop_prob)
        
        self.linear_global = nn.Linear(embed_size, num_heads)
        
        self.pos_key_embeds = nn.Embedding(max_pos, self.head_size)
        self.pos_value_embeds = nn.Embedding(max_pos, self.head_size)
        
        self.linear_layers = nn.ModuleList([
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size),
            nn.Linear(embed_size, embed_size)
        ])
        
        self.linear_global = nn.Linear(embed_size, num_heads)
        self.dropout = nn.Dropout(drop_prob)
    
    def forward(self, query, item_ids, skill_ids):
        """
        Forward pass
        
        Args:
            query: input tensor [batch_size, seq_len, embed_size]
            item_ids: item ids [batch_size, seq_len]
            skill_ids: skill ids [batch_size, seq_len]
        
        Returns:
            output [batch_size, seq_len, embed_size]
            attention_weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = query.shape
        
        query = self.linear_q(query)
        key = self.linear_k(query)
        value = self.linear_v(query)
        
        score_glo = self.linear_global(key).unsqueeze(-1).transpose(1, 2)
        
        query = query.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # 简化位置编码：使用绝对位置
        pos_ids = torch.arange(seq_len, device=query.device)
        # 限制位置范围在 [0, max_pos-1]
        pos_ids = torch.clamp(pos_ids, 0, self.max_pos - 1)
        # 获取位置嵌入 [seq_len, head_size]
        pos_embeds = self.pos_key_embeds(pos_ids)
        # 扩展维度 [1, 1, seq_len, head_size]
        pos_embeds = pos_embeds.unsqueeze(0).unsqueeze(0)
        # 计算位置分数 [batch_size, num_heads, seq_len, seq_len]
        pos_scores = torch.matmul(query, pos_embeds.transpose(-2, -1))
        
        scores = scores + score_glo + pos_scores
        
        scores = scores / math.sqrt(self.head_size)
        
        prob_attn = F.softmax(scores, dim=-1)
        
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        
        output = torch.matmul(prob_attn, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_size)
        
        for linear_layer in self.linear_layers:
            output = linear_layer(output)
            output = F.gelu(output)
        
        return output, prob_attn