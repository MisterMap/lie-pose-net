import torch
import torch.nn as nn


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hidden_dim, heads_count):
        super().__init__()

        assert hidden_dim % heads_count == 0, "hid_dim must be divisible by n_heads"

        self._hidden_dim = hidden_dim
        self._heads_count = heads_count
        self._head_dim = hidden_dim // heads_count

        # query, key and value linear networks
        self._query_linear_module = nn.Linear(hidden_dim, hidden_dim)
        self._key_linear_module = nn.Linear(hidden_dim, hidden_dim)
        self._value_linear_module = nn.Linear(hidden_dim, hidden_dim)

        # output linear networks
        self._output_linear_module = nn.Linear(hidden_dim, hidden_dim)

        # scale parameter
        self._scale = torch.nn.Parameter(torch.sqrt(torch.tensor(self._head_dim, dtype=torch.float32)),
                                         requires_grad=False)

    def forward(self, query, key=None, value=None, mask=None):
        """
        query/key/value (batch of queries/keys/values): torch.float32 tensor of shape [bs, seq_len, hid_dim]
        mask (mask of valid elements): torch.bool tensor of shape [bs, seq_len]
        returns (multi-head attention): torch.float32 tensor of shape [bs, seq_len, hid_dim]
        """
        if key is None or value is None:
            key = query
            value = query

        if query.dim() == 2:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        input_tensor = query
        batch_size, sequence_length = query.shape[0], query.shape[1]

        # calculate Q, K, V using corresponding linear networks
        query, key, value = self._query_linear_module(query), self._key_linear_module(key), self._value_linear_module(
            value)  # shape is [bs, seq_len, hid_dim]

        # prepare Q, K, V for .matmul() or `@` operator
        # shape is [bs, n_heads, seq_len, head_dim]
        query = query.view(batch_size, sequence_length, self._heads_count, self._head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self._heads_count, self._head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self._heads_count, self._head_dim).transpose(1, 2)

        # compute energy using .matmul() or `@` operator (don't forget to scale!)
        # shape is [bs, n_heads, seq_len, seq_len]
        energy = query @ key.transpose(2, 3) / self._scale

        # apply mask – 1 in mask is a valid element, 0 - not (use .masked_fill())
        if mask is not None:
            energy = energy * mask[:, None, :, None] * mask[:, None, None, :]

        # apply softmax along the last dim of energy and get the attention weights
        # shape is [bs, n_heads, seq_len, seq_len]
        attention = torch.softmax(energy, dim=2)

        # weight values with calculated attention (use .matmul() or `@` operator)
        # shape is [bs, n_heads, seq_len, head_dim]
        x = attention @ value

        # squash 1 and 4 dims back
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self._hidden_dim)  # shape is [bs, seq_len, hid_dim]

        # apply output linear layer
        x = self._output_linear_module(x)

        return x.squeeze()
