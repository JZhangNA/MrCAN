import torch.nn as nn
import torch
from networks.SubLayers import MultiHead_VariAttention, PositionwiseFeedForward, MultiHead_TempAttention
import torch.nn.functional as F

class EncoderLayer(nn.Module):
    def __init__(self, n_multiv, d_model, window, d_inner, n_head, d_k, d_v, dropout=0.1):
        super().__init__()

        self.window = window
        self.n_multiv = n_multiv
        self.dropout = dropout
        local = 4
        self.conv2 = nn.Conv2d(1, d_model, (local, 1))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, n_multiv+1))
        self.dropout = nn.Dropout(dropout)
        self.slf_attn = MultiHead_VariAttention(
            n_head, d_model, window, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(window, window * 2, dropout=dropout)

    def forward(self, enc_input):
        v = torch.transpose(enc_input, 1, 2)
        x = enc_input.view(-1, 1, self.window, self.n_multiv+1)
        x = F.relu(self.conv2(x))
        x = self.pooling1(x)
        x = self.dropout(x)
        x = torch.squeeze(x, 2)
        x = torch.transpose(x, 1, 2)
        q = k = x
        enc_output, enc_slf_attn = self.slf_attn(q, k, v)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, filter_size, dropout=0.1):
        super().__init__()
        self.slf_attn = MultiHead_TempAttention(n_head, d_model, d_k, d_v, filter_size=filter_size, dropout=dropout)
        self.enc_attn = MultiHead_TempAttention(n_head, d_model, d_k, d_v, filter_size=filter_size, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_input):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_input, enc_input)
        dec_output = self.pos_ffn(dec_output)
        return dec_output
