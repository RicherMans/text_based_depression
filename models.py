import torch
import torch.nn as nn


def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)


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
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.LSTM(inputdim, batch_first=True, **kwargs)
        init_rnn(self.net)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(
            rnn_outputdim[-1], output_size)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
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
        return self.outputlayer(x)


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
        self.attn = SimpleAttention(kwargs['hidden_size']*2)

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
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.lstm = LSTM(inputdim, output_size, **kwargs)
        init_rnn(self.lstm)
        self.attn = SimpleAttention(kwargs['hidden_size']*2)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        x, _ = self.lstm.net(x)
        x = self.attn(x)[0].unsqueeze(1)
        return self.lstm.outputlayer(x)

    def extract_feature(self, x):
        x, _ = self.lstm.net(x)
        return self.attn(x)[1] * x


class SimpleAttention(nn.Module):

    """Docstring for SimpleAttention. """

    def __init__(self, inputdim):
        """TODO: to be defined1.

        :inputdim: TODO

        """
        nn.Module.__init__(self)

        self._inputdim = inputdim
        self.attn = nn.Linear(inputdim, 1, bias=False)
        nn.init.xavier_uniform_(self.attn.weight)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        out = torch.bmm(weights.transpose(1, 2), torch.tanh(x)).squeeze(0)
        return out, weights
