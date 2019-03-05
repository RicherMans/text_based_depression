import torch
import torch.nn as nn
from TCN.tcn import TemporalConvNet


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
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.LSTM(inputdim, batch_first=True, **kwargs)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(rnn_outputdim[-1], output_size)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        x, _ = self.net(x)
        return self.outputlayer(x)


class DNN(torch.nn.Module):

    """DNN class for Depression detection"""

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of DNN
        :**kwargs: Other args, passed down to nn.DNN


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('layers', [512]*6)
        kwargs.setdefault('dropout', 0.5)
        layers = [inputdim] + kwargs['layers']
        net = nn.ModuleList()
        for h0, h1 in zip(layers, layers[1:]):
            net.append(nn.Linear(h0, h1))
            net.append(nn.ReLU(True))
            if kwargs['dropout'] > 0:
                net.append(nn.Dropout(kwargs['dropout']))
        self.net = nn.Sequential(*net)
        net_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(net_outputdim[-1], output_size)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        x = self.net(x)
        return self.outputlayer(x)


class TCN(nn.Module):
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(TCN, self).__init__()
        kernel_size = kwargs.get('kernel_size', 3)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.linear(x)
