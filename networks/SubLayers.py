import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from networks import causal_convolution_layer


class MultiHead_VariAttention(nn.Module):
    def __init__(self, n_head, d_model, window, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(window, n_head * d_v)

        self.fc = nn.Linear(n_head * d_v, window)

        self.layer_norm = nn.LayerNorm(window)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = v
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(d_k)
        scores = self.dropout1(F.softmax(attn, dim = -1))
        output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(sz_b, -1, n_head * d_v)
        output = self.fc(output)
        output = self.dropout2(F.relu(output))

        output = self.layer_norm(output + residual)

        return output, scores

class MultiHead_TempAttention(nn.Module):

    def __init__(self, n_head, d_model, d_k, d_v, filter_size, dropout):

        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.conv_q = causal_convolution_layer.context_embedding(in_channels=d_model, embedding_size=d_k * n_head, k=filter_size)
        self.conv_k = causal_convolution_layer.context_embedding(in_channels=d_model, embedding_size=d_k * n_head, k=filter_size)
        self.dense_v = nn.Linear(d_model, d_v * n_head)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        residual = q

        q = torch.transpose(q, 1, 2)
        k = torch.transpose(k, 1, 2)

        sz_b, _, time_step = q.size()
        sz_b, _, time_step = k.size()
        sz_b, time_step, _ = v.size()

        q = self.conv_q(q).view(sz_b, d_k, n_head, time_step)
        k = self.conv_k(k).view(sz_b, d_k, n_head, time_step)
        v = self.dense_v(v).view(sz_b, time_step, d_v, n_head)

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 3, 1)
        v = v.permute(0, 3, 1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn / math.sqrt(d_k)
        scores = F.softmax(attn, dim=-1)
        scores = self.dropout1(scores)
        output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(sz_b, -1, n_head * d_v) 
        output = self.fc(output)

        output = F.relu(output)
        output = self.dropout2(output)
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(self.dropout1(F.relu(self.w_1(output))))
        output = self.dropout2(F.relu(output))
        output = output.transpose(1, 2)
        output = self.layer_norm(output + residual)
        return output
























