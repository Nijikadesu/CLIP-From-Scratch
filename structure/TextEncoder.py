import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from data.data_config import get_word_dict

# Implementation of Text Decoder, using Transformer as Backbone.
# Text template: { a photo of number X }
def scaled_dot_product(q, k, v):
    d_k = q.shape[-1]
    scaled = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self attention layer.
    """
    def __init__(self, d_model=16, num_heads=4):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.qkv_layer(x)
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.chunk(3, dim=-1)
        values, _ = scaled_dot_product(q, k, v)
        values = values.reshape(batch_size, seq_len, self.d_model)
        output = self.linear_layer(values)
        return output

class FeedForwardNetwork(nn.Module):
    """
    Feed forward network.
    """
    def __init__(self, d_model=16, hidden_dim=32, drop_prob=0.1):
        super().__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(F.relu(x))
        x = self.linear2(x)
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding module.
    """
    def __init__(self, d_model=16, max_seq_len=19):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_seq_len).reshape(self.max_seq_len, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=-1)
        PE = torch.flatten(stacked, start_dim=1, end_dim=-1)
        return PE

class Tokenizer(nn.Module):
    """
    Tokenizer。
    """
    def __init__(self, word_dict):
        super().__init__()
        self.word_dict = word_dict
        self.embedding = nn.Embedding(num_embeddings=len(word_dict), embedding_dim=16)
        self.positional_encoding = PositionalEncoding(d_model=16, max_seq_len=19)

    def forward(self, batch):
        batch_list = []
        for sentence in batch:
            word_list = []
            for ch in list(sentence):
                word_list.append(self.word_dict[ch])
            batch_list.append(word_list)
        word_tensor = torch.Tensor(batch_list).long()
        embedding = self.embedding(word_tensor)
        embedding += self.positional_encoding()
        return embedding

class TextEncoder(nn.Module):
    """
    text encoder.
    """
    def __init__(self, d_model=16, word_dict=None, drop_prob=0.1):
        super().__init__()
        self.word_dict = word_dict
        self.tokenizer = Tokenizer(word_dict)
        self.self_attention = MultiheadSelfAttention(d_model=16, num_heads=4)
        self.ffn = FeedForwardNetwork(d_model=16, hidden_dim=32, drop_prob=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.linear1 = nn.Linear(19 * 16, 16)
        self.linear2 = nn.Linear(16, 8)

    def forward(self, x):
        x = self.tokenizer(x)
        print(x.shape)

        residual_x = x
        x = self.self_attention(x)
        x = self.dropout1(x)
        x = self.norm1(x + residual_x)

        residual_x = x
        x = self.ffn(x)
        x = self.dropout2(x)
        x = self.norm2(x + residual_x)

        x = x.reshape(-1, 19 * 16)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x

if __name__ == '__main__':
    x = ['a photo of number 3', 'a photo of number 4']
    word_dict = get_word_dict()
    encoder = TextEncoder(word_dict=word_dict)
    print(encoder)
    output = encoder(x)
    print(output.shape)