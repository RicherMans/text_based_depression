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


class CosineLoss(torch.nn.Module):
    """description"""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.norm = torch.nn.functional.normalize

    @staticmethod
    def label_to_onehot(tar, nlabels=2):
        if tar.ndimension() == 1:
            tar = tar.unsqueeze(-1)  # add singleton [B, 1]
        tar_onehot = tar.new_zeros((len(tar), nlabels)).detach()
        tar_onehot.scatter_(1, tar.long(), 1)
        return tar_onehot.float()

    def forward(self, input, target):
        target = CosineLoss.label_to_onehot(target)
        if input.ndimension() == 2:
            input = input.unsqueeze(-1)  # add singleton dimension
        if target.ndimension() == 2:
            target = target.unsqueeze(1)  # Add singleton dimension
        norm_input = self.norm(input, p=2, dim=1)
        #Input shape: [Bx1xC]
        #Target shape: [BxCx1]
        cos_loss = 1 - torch.bmm(target, norm_input)
        return cos_loss.mean()


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
        return self.loss(input, target.float())


class DepressionLoss(torch.nn.Module):
    """Docstring for DepressionLoss. """
    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = MAELoss()
        self.bin_loss = BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.score_loss(input[:, 0], target[:, 0]) + self.bin_loss(
            input[:, 1], target[:, 1])


class DepressionLossSmoothCos(torch.nn.Module):
    """Docstring for DepressionLoss. """
    def __init__(self):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = HuberLoss()
        self.cos_loss = CosineLoss()

    def forward(self, input, target):
        target = target.long()
        phq8_pred, phq8_tar = input[:, 0], target[:, 0]
        binary_pred, binary_tar = input[:, 1:3], target[:, 1]
        return self.score_loss(phq8_pred, phq8_tar) + self.cos_loss(
            binary_pred, binary_tar)


class DepressionLossSmooth(torch.nn.Module):
    """Docstring for DepressionLoss. """
    def __init__(self, reduction='sum'):
        """ """
        torch.nn.Module.__init__(self)

        self.score_loss = HuberLoss()
        self.bce = BCEWithLogitsLoss()
        self.weight = torch.nn.Parameter(torch.tensor(0.))
        self.reduction = reduction
        self.eps = 0.01

    def forward(self, input, target):
        phq8_pred, phq8_tar = input[:, 0], target[:, 0]
        binary_pred, binary_tar = input[:, 1], target[:, 1]
        score_loss, bin_loss = self.score_loss(phq8_pred, phq8_tar), self.bce(
            binary_pred, binary_tar)
        weight = torch.clamp(torch.sigmoid(self.weight),
                             min=self.eps,
                             max=1 - self.eps)
        stacked_loss = (weight * score_loss) + ((1 - weight) * bin_loss)
        if self.reduction == 'mean':
            stacked_loss = stacked_loss.mean()
        elif self.reduction == 'sum':
            stacked_loss = stacked_loss.sum()
        return stacked_loss


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
