import torch
import math
import torch.nn as nn
from TCN.tcn import TemporalConvNet


def init_rnn(rnn):
    """init_rnn
    Initialized RNN weights, independent of type GRU/LSTM/RNN

    :param rnn: the rnn model 
    """
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


class AutoEncoderLSTM(nn.Module):
    """docstring for AutoEncoderLSTM"""
    def __init__(self, inputdim, output_size=None, **kwargs):
        super(AutoEncoderLSTM, self).__init__()
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 3)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.LSTM(inputdim, batch_first=True, **kwargs)
        self.decoder = nn.LSTM(input_size=kwargs['hidden_size'] *
                               (int(kwargs['bidirectional']) + 1),
                               hidden_size=inputdim,
                               batch_first=True,
                               bidirectional=kwargs['bidirectional'])
        self.squeezer = nn.Sequential()
        if kwargs['bidirectional']:
            self.squeezer = nn.Linear(inputdim * 2, inputdim)

    def forward(self, x):
        enc_o, _ = self.net(x)
        out, _ = self.decoder(enc_o)
        return self.squeezer(out)


class LSTM(torch.nn.Module):
    """LSTM class for Depression detection"""
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of LSTM
        :**kwargs: Other args, passed down to nn.LSTM


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 2)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.1)
        self.net = nn.LSTM(inputdim, batch_first=True, **kwargs)
        init_rnn(self.net)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(rnn_outputdim[-1], output_size)

    def forward(self, x: torch.tensor):
        """Forwards input vector through network

        :x: torch.tensor
        :returns: TODO

        """
        x, _ = self.net(x)
        return self.outputlayer(x)


class GRU(torch.nn.Module):
    """GRU class for Depression detection"""
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of GRU
        :**kwargs: Other args, passed down to nn.GRU


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.GRU(inputdim, batch_first=True, **kwargs)
        init_rnn(self.net)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(rnn_outputdim[-1], output_size)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        x, _ = self.net(x)
        return self.outputlayer(x)  # Pool time


class GRUAttn(torch.nn.Module):
    """GRUAttn class for Depression detection"""
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of GRUAttn
        :**kwargs: Other args, passed down to nn.GRUAttn


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.GRU(inputdim, batch_first=True, **kwargs)
        init_rnn(self.net)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(rnn_outputdim[-1], output_size)
        self.attn = SimpleAttention(kwargs['hidden_size'] * 2)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        x, _ = self.net(x)
        x = self.attn(x)[0].unsqueeze(1)
        return self.outputlayer(x)


class LSTMAttn(torch.nn.Module):
    """LSTMSimpleAttn class for Depression detection"""
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of LSTMSimpleAttn
        :**kwargs: Other args, passed down to nn.LSTMSimpleAttn


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 3)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.lstm = LSTM(inputdim, output_size, **kwargs)
        init_rnn(self.lstm)
        self.attn = SimpleAttention(kwargs['hidden_size'] * 2)

    def forward(self, x):
        """Forwards input vector through network

        :x: input tensor of shape (B, T, D) [Batch, Time, Dim]
        :returns: TODO

        """
        x, _ = self.lstm.net(x)
        x = self.attn(x)[0]
        return self.lstm.outputlayer(x)

    def extract_feature(self, x):
        x, _ = self.lstm.net(x)
        return self.attn(x)[1] * x


class LSTMSelfAttn(nn.Module):
    """docstring for LSTMSelfAttn"""
    def __init__(self, inputdim, output_size, **kwargs):
        super(LSTMSelfAttn, self).__init__()
        self.inputdim, output_size, = inputdim, output_size
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.1)
        self.lstm = LSTM(inputdim, output_size, **kwargs)
        init_rnn(self.lstm)
        self.attn = SelfAttention()

    def forward(self, x):
        """Forwards input vector through network

        :x: input tensor of shape (B, T, D) [Batch, Time, Dim]
        :returns: TODO

        """
        out, hid = self.lstm.net(x)
        hid = hid[1]
        # Bidirectional
        hid = torch.cat([hid[-1], hid[-2]], dim=1)
        # 0 for LSTM
        energy, out = self.attn(hid, out, out)
        out = out.unsqueeze(1)
        return self.lstm.outputlayer(out)


class SelfAttention(nn.Module):
    """docstring for SelfAttention"""
    def __init__(self):
        super(SelfAttention, self).__init__()

    def forward(self, query, keys, values):
        query = query.unsqueeze(1)
        keys = keys.transpose(1, 2)  # [BxTxK] --> [BxKxT]
        energy = torch.bmm(query, keys)  # [Bx1xT]
        energy = torch.softmax(energy.mul_(1. / math.sqrt(query.shape[-1])),
                               dim=-1)

        comb = torch.bmm(energy, values).squeeze(1)  # [BxV]
        return energy, comb


class SimpleAttention(nn.Module):
    """Docstring for SimpleAttention. """
    def __init__(self, inputdim):
        """TODO: to be defined1.

        :inputdim: TODO

        """
        nn.Module.__init__(self)

        self._inputdim = inputdim
        self.attn = nn.Linear(inputdim, 1, bias=False)
        nn.init.kaiming_normal_(self.attn.weight)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        out = torch.bmm(weights.transpose(1, 2), torch.tanh(x))
        return out, weights
