import torch
import torch.nn as nn
from TCN.tcn import TemporalConvNet


def init_rnn(rnn):
    for name, param in rnn.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param)
            # Init forgetgate with 1
            n = param.shape[0]
            nn.init.constant_(param[n // 4:n // 2], 1.0)


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
        nn.init.xavier_uniform_(self.outputlayer.weight)
        nn.init.constant_(self.outputlayer.bias, 0.)

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


class GRUPosattn(torch.nn.Module):

    """GRUPosattn class for Depression detection"""

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of GRUPosattn
        :**kwargs: Other args, passed down to nn.GRUPosattn


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.net = nn.GRU(inputdim, batch_first=True, **kwargs)
        rnn_outputdim = self.net(torch.randn(1, 50, inputdim))[0].shape
        self.outputlayer = nn.Linear(rnn_outputdim[-1], output_size)
        self.attn = nn.Linear(inputdim, 1)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        pos_attn = torch.sigmoid(self.attn(x))
        x, _ = self.net(x)
        x = pos_attn*x
        return self.outputlayer(x)


class LSTMSimpleAttn(torch.nn.Module):

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


class LSTMAttn(torch.nn.Module):

    """LSTMAttn class for Depression detection"""

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of LSTMAttn
        :**kwargs: Other args, passed down to nn.LSTMAttn


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.lstm = LSTM(inputdim, output_size, **kwargs)
        self.local_embed = nn.Linear(inputdim, 1)
        self.attn = SimpleAttention(kwargs['hidden_size']*2)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        lx = torch.sigmoid(self.local_embed(x))
        x, _ = self.lstm.net(x)
        x = lx*x
        x = self.attn(x)[0].unsqueeze(1)
        return self.lstm.outputlayer(x)


class LSTMPosAttn(torch.nn.Module):

    """LSTMPosAttn class for Depression detection"""

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of LSTMPosAttn
        :**kwargs: Other args, passed down to nn.LSTMPosAttn


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('hidden_size', 128)
        kwargs.setdefault('num_layers', 4)
        kwargs.setdefault('bidirectional', True)
        kwargs.setdefault('dropout', 0.2)
        self.lstm = LSTM(inputdim, output_size, **kwargs)
        self.local_embed = nn.Linear(inputdim, 1)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        lx = torch.sigmoid(self.local_embed(x))
        x, _ = self.lstm.net(x)
        x = lx*x
        return self.lstm.outputlayer(x)


class DNN(torch.nn.Module):

    """DNN class for Depression detection"""

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        """

        :inputdim:int: Input dimension
        :output_size:int: Output dimension of DNN
        :**kwargs: Other args, passed down to nn.DNN


        """
        torch.nn.Module.__init__(self)
        kwargs.setdefault('layers', [128]*10)
        kwargs.setdefault('dropout', 0.2)
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
        self.pos_attn = nn.Linear(inputdim, 1)

    def forward(self, x):
        """Forwards input vector through network

        :x: TODO
        :returns: TODO

        """
        pos_attn = torch.sigmoid(self.pos_attn(x))
        x = self.net(x)
        x = pos_attn*x
        return self.outputlayer(x)


class TCN(nn.Module):
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(TCN, self).__init__()
        kernel_size = kwargs.get('kernel_size', 5)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        return self.linear(x)


class BTCN(nn.Module):
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(BTCN, self).__init__()
        kernel_size = kwargs.get('kernel_size', 3)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.forward_tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.backward_tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(2*num_channels[-1], output_size)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""

        forward_x = self.forward_tcn(x.transpose(1, 2)).transpose(1, 2)
        flipped_x = torch.flip(x, [1])
        backward_x = self.backward_tcn(
            flipped_x.transpose(1, 2)).transpose(1, 2)
        return self.linear(torch.cat((forward_x, backward_x), dim=-1))


class TCNPosAttn(nn.Module):

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(TCNPosAttn, self).__init__()
        kernel_size = kwargs.get('kernel_size', 5)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.position_attention = nn.Linear(inputdim, 1)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        pos_attn = torch.sigmoid(self.position_attention(x))
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = pos_attn*x
        return self.linear(x)


class TCNAttn(nn.Module):

    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(TCNAttn, self).__init__()
        kernel_size = kwargs.get('kernel_size', 5)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.attn = SimpleAttention(num_channels[-1])

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x = self.attn(x)[0].unsqueeze(1)
        return self.linear(x)


class BTCNAttn(nn.Module):
    def __init__(self, inputdim: int, output_size: int, **kwargs):
        super(BTCNAttn, self).__init__()
        kernel_size = kwargs.get('kernel_size', 3)
        dropout = kwargs.get('dropout', 0.05)
        num_channels = kwargs.get('num_channels', [128]*5)
        self.forward_tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.backward_tcn = TemporalConvNet(
            inputdim, num_channels, kernel_size=kernel_size, dropout=dropout)
        # self.self_attn = SelfAttention(
        # num_channels[-1]*2, num_channels[-1]*2, num_channels[-1]*2)
        # self.attn = SimpleAttention(2*num_channels[-1])
        self.linear = nn.Linear(2*num_channels[-1], output_size)
        self.forward_position = nn.Linear(inputdim, 1)
        self.backward_position = nn.Linear(inputdim, 1)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        fwd_position_attn = torch.sigmoid(self.forward_position(x))
        forward_x = self.forward_tcn(x.transpose(1, 2)).transpose(1, 2)
        forward_x = fwd_position_attn*forward_x
        flipped_x = torch.flip(x, [1])
        bwd_position_attn = torch.sigmoid(self.backward_position(flipped_x))
        backward_x = self.backward_tcn(
            flipped_x.transpose(1, 2)).transpose(1, 2)
        backward_x = backward_x*bwd_position_attn
        fwd_bwd = torch.cat((forward_x, backward_x), dim=-1)
        # out = self.self_attn(fwd_bwd)
        # out = self.attn(fwd_bwd)[0].unsqueeze(1)
        return self.linear(fwd_bwd)


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


class SelfAttention(nn.Module):
    def __init__(self, inputdim, k_size, v_size):
        super(SelfAttention, self).__init__()
        self.key_layer = nn.Linear(inputdim, k_size)
        self.query_layer = nn.Linear(inputdim, k_size)
        self.value_layer = nn.Linear(inputdim, v_size)
        self.scaler = nn.LayerNorm(k_size)
        self.sq_k = torch.sqrt(torch.tensor(k_size).float())

    def forward(self, x):
        keys = self.key_layer(x)
        queries = self.query_layer(x)
        values = self.value_layer(x)
        logits = torch.bmm(queries, keys.transpose(2, 1))
        # Do not mask
        q_k = torch.softmax(logits, dim=1) / self.sq_k
        applied = torch.bmm(q_k, values)
        return self.scaler(applied + x)


class BiGRU(nn.Module):
    """BiGRU"""

    def __init__(self, inputdim, outputdim, bidirectional=True, **kwargs):
        nn.Module.__init__(self)

        self.rnn = nn.GRU(inputdim, outputdim,
                          bidirectional=bidirectional, batch_first=True, **kwargs)

    def forward(self, x, hid=None):
        x, hid = self.rnn(x)
        return x, (hid,)


class StandardBlock(nn.Module):

    """docstring for StandardBlock"""

    def __init__(self, inputfilter, outputfilter, kernel_size, stride, padding, bn=True, **kwargs):
        super(StandardBlock, self).__init__()
        self.activation = kwargs.get('activation', nn.ReLU(True))
        self.batchnorm = nn.Sequential() if not bn else nn.BatchNorm2d(inputfilter)
        if self.activation.__class__.__name__ == 'GLU':
            outputfilter = outputfilter * 2
        self.conv = nn.Conv2d(inputfilter, outputfilter,
                              kernel_size=kernel_size, stride=stride, bias=not bn, padding=padding)

    def forward(self, x):
        x = self.batchnorm(x)
        x = self.conv(x)
        return self.activation(x)


class CRNN(nn.Module):

    """Encodes the given input into a fixed sized dimension"""

    def __init__(self, inputdim, output_size, **kwargs):
        super(CRNN, self).__init__()
        self._inputdim = inputdim
        self._embed_size = output_size
        self._filtersizes = kwargs.get(
            'filtersizes', [3, 3, 3, 3, 3])
        self._filter = kwargs.get(
            'filter', [16, 32, 128, 128, 128])
        self._pooling = kwargs.get(
            'pooling', [(1, 4), (1, 3), (1, 2), (1, 2), (1, 2)])
        self._hidden_size = kwargs.get('hidden_size', 128)
        self._bidirectional = kwargs.get('bidirectional', True)
        self._rnn = kwargs.get('rnn', 'BiGRU')
        self._activation = kwargs.get('activation', 'ReLU')
        self._blocktype = kwargs.get('blocktype', 'StandardBlock')
        self._bn = kwargs.get('bn', True)

        self._filter = [1] + self._filter
        net = nn.ModuleList()
        assert len(self._filter) - 1 == len(self._filtersizes)
        for nl, (h0, h1, filtersize, poolingsize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes, self._pooling)):
            # Stop in zip_longest when last element arrived
            if not h1:
                break
            current_activation = getattr(nn, self._activation)(True)
            net.append(globals()[self._blocktype](
                inputfilter=h0, outputfilter=h1, kernel_size=filtersize, padding=int(filtersize)//2, bn=self._bn, stride=1, activation=current_activation))
            # Poolingsize will be None if pooling is finished
            if poolingsize:
                net.append(nn.MaxPool2d(poolingsize))
            # Only dropout at last layer before GRU
            if nl == (len(self._filter) - 2):
                net.append(nn.Dropout(0.1))
        self.network = nn.Sequential(*net)

        def calculate_cnn_size(input_size):
            x = torch.randn(input_size).unsqueeze(0)
            output = self.network(x)
            return output.size()[1:]
        outputdim = calculate_cnn_size((1, 500, inputdim))
        self.rnn = globals()[self._rnn](
            self._filter[-1] * outputdim[-1], self._hidden_size, self._bidirectional)
        rnn_output = self.rnn(torch.randn(
            1, 500, self._filter[-1]*outputdim[-1]))[0].shape[-1]
        self.outputlayer = nn.Linear(
            rnn_output,
            self._embed_size)
        self.network.apply(self.init_weights)
        self.outputlayer.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight)
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        # Add dimension for filters
        x = x.unsqueeze(1)
        x = self.network(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1)
        x, _ = self.rnn(x)
        x = self.outputlayer(x)
        return x
