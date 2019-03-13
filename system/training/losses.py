import torch

CrossEntropyLoss = torch.nn.CrossEntropyLoss


class MAELoss(torch.nn.Module):

    """Docstring for MAELoss. """

    def __init__(self):
        """TODO: to be defined1. """
        torch.nn.Module.__init__(self)
        self.loss = torch.nn.L1Loss()

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        return self.loss(input, target)


class RMSELoss(torch.nn.Module):

    """Docstring for RMSELoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.loss = torch.nn.MSELoss()

    def forward(self, input, target):
        target = target.float()
        return torch.sqrt(self.loss(input, target))


class MSELoss(torch.nn.Module):

    """Docstring for MSELoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.loss = torch.nn.MSELoss()

    def forward(self, input, target):
        target = target.float()
        return self.loss(input, target)


class HuberLoss(torch.nn.Module):

    """Docstring for HuberLoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, input, target):
        target = target.float()
        return self.loss(input, target)


class DepressionLoss(torch.nn.Module):

    """Docstring for DepressionLoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = MAELoss()
        self.bin_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.score_loss(input[:, 0], target[:, 0]) + self.bin_loss(input[:, 1], target[:, 1])


class DepressionLossMSE(torch.nn.Module):

    """Docstring for DepressionLoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = MSELoss()
        self.bin_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        score_loss = self.score_loss(input[:, 0], target[:, 0])
        binary_loss = self.bin_loss(input[:, 1], target[:, 1])
        return score_loss + binary_loss


class DepressionLossSmooth(torch.nn.Module):

    """Docstring for DepressionLoss. """

    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = HuberLoss()
        self.bin_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.score_loss(input[:, 0], target[:, 0]) + self.bin_loss(input[:, 1], target[:, 1])


class BCEWithLogitsLoss(torch.nn.Module):

    """Docstring for BCEWithLogitsLoss. """

    def __init__(self):
        """TODO: to be defined1. """
        torch.nn.Module.__init__(self)

        self.loss = torch.nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        input = input.float()
        target = target.float()
        return self.loss(input, target)
