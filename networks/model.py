import torch.nn as nn
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from networks.Layers import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self,
            window, n_multiv,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, drop_prob=0.1):

        super().__init__()

        self.n_head = n_head
        self.n_layers = n_layers
        self.window = window
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob

        self.layer_stack = nn.ModuleList([
            EncoderLayer(n_multiv=n_multiv, d_model=d_model, window=window, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=drop_prob)
            for _ in range(n_layers)])

    def forward(self, x):
        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            enc_output = torch.transpose(enc_output, 1, 2)
        return enc_output

class Decoder(nn.Module):
    def __init__(
            self,
            window, n_multiv,
            d_k, d_v, d_model, d_inner,
            n_layers, n_head, filter_size, drop_prob=0.1):
        super().__init__()

        self.window = window
        self.n_multiv = n_multiv
        self.d_model = d_model
        self.drop_prob = drop_prob
        self.n_head = n_head

        self.linear2 = nn.Linear(1, d_model)
        local = 4
        self.conv2 = nn.Conv2d(1, d_model, (local, 1))
        self.pooling1 = nn.AdaptiveMaxPool2d((1, window))

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, filter_size=filter_size, dropout=drop_prob)
            for _ in range(n_layers)])
        self.dropout1 = nn.Dropout(p=self.drop_prob)

    def forward(self, x, y_prev):
        x = torch.transpose(x, 1, 2)
        x = x.view(-1, 1, self.n_multiv+1, self.window)
        x = F.relu(self.conv2(x))
        x = self.pooling1(x)
        x = self.dropout1(x)
        x = torch.squeeze(x, 2)
        enc_input = torch.transpose(x, 1, 2)
        dec_output = self.linear2(y_prev)
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, enc_input)

        return dec_output

class BatchAttention(nn.Module):
    def __init__(self, add_bf=1, dim=160):
        super(BatchAttention, self).__init__()
        self.encoder_layers = torch.nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=dim, dropout=0.2)

    def forward(self, x):
        if self.training:
            # print('[training] add BatchAttention!')
            o_x = torch.cat([x, self.encoder_layers(x)], dim=0) if x is not None else None
            return o_x
        return x

class MrCAN(nn.Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.window = hparams.window
        self.n_multiv = hparams.n_multiv
        self.d_model = hparams.d_model
        self.d_k = hparams.d_k
        self.d_v = hparams.d_v
        self.d_inner = hparams.d_inner
        self.n_layers = hparams.n_layers
        self.n_head = hparams.n_head
        self.filter_size = hparams.filter_size
        self.drop_prob = hparams.drop_prob
        self.add_bf = hparams.add_bf
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        self.encoder = Encoder(
            window=self.window, n_multiv=self.n_multiv, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, drop_prob=self.drop_prob)

        self.decoder = Decoder(
            window=self.window, n_multiv=self.n_multiv, d_k=self.d_k, d_v=self.d_v, d_model=self.d_model,
            d_inner=self.d_inner, n_layers=self.n_layers, n_head=self.n_head, filter_size=self.filter_size, drop_prob=self.drop_prob)

        if self.add_bf == 1:
            self.SRT = BatchAttention(self.add_bf, self.d_model)

        self.fc1 = nn.Linear(self.d_model, 1)

    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, x, y_prev):
        x1 = torch.cat((x, y_prev), dim=2)
        x = self.encoder(x1)
        output = self.decoder(x, y_prev)
        output = output[:,-1,:]
        if self.training and self.add_bf:
            output = self.SRT(output)
            bs = int(len(output) /2)
            output1 = output[:bs]
            output1 = self.fc1(output1.reshape(bs,-1))
            output2 = output[bs:]
            output2 = self.fc1(output2.reshape(bs,-1))
            return output1,output2
        else:
            bs = int(len(output))
            output = self.fc1(output.reshape(bs,-1))
            return output, None

    @staticmethod
    def add_model_specific_args(filter_size=9):
        parser = ArgumentParser(description='PyTorch Time series forecasting')
        parser.add_argument('--d_model', type=int, default=160)
        parser.add_argument('--d_inner', type=int, default=320)
        parser.add_argument('--d_k', type=int, default=40)
        parser.add_argument('--d_v', type=int, default=40)
        parser.add_argument('--n_head', type=int, default=4)
        parser.add_argument('--n_layers', type=int, default=3)
        parser.add_argument('--filter_size', type=int, default=filter_size)
        parser.add_argument('--drop_prob', type=float, default=0.2)
        parser.add_argument('--n_multiv', type=int, default=18)  # dimension of driving series
        parser.add_argument('--window', default=64, type=int)
        parser.add_argument('--add_bf', default=1, type=int)

        return parser
















