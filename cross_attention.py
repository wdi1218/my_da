import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CrossAttention, self).__init__()
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # 线性变换
        query = self.query_proj(query)
        key = self.key_proj(key)
        value = self.value_proj(value)

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_probs = self.softmax(attention_scores)

        # 加权求和
        attention_output = torch.matmul(attention_probs, value)
        return attention_output