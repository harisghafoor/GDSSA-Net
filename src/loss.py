import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')

        Dice_BCE = BCE + dice_loss

        return Dice_BCE

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    union = torch.sum(score + target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - score

    return loss, score

class ValidationDiceLoss(nn.Module):
    def __init__(self):
        super(ValidationDiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-5):
        inputs = torch.sigmoid(inputs)
        inputs = (inputs > 0.5 ).float()
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice

class HausdorffDTLoss(nn.Module):
    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = distance_transform_edt(fg_mask)
                bg_dist = distance_transform_edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of the same dimension"

        device = pred.device
        target = target.to(device)

        pred_dt = torch.from_numpy(self.distance_field(pred.detach().cpu().numpy())).to(device).float()
        target_dt = torch.from_numpy(self.distance_field(target.detach().cpu().numpy())).to(device).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        return loss

class DiceLoss_V1(nn.Module):
    def __init__(self):
        super(DiceLoss_V1, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        total_loss = 0

        for i in range(inputs.size(0)):
            input_i = inputs[i].view(-1)
            target_i = targets[i].view(-1)

            intersection = (input_i * target_i).sum()
            dice = (2. * intersection + smooth) / (input_i.sum() + target_i.sum() + smooth)

            total_loss += 1 - dice

        average_loss = total_loss / inputs.size(0)

        return average_loss

class DiceTestingScore(nn.Module):
    def __init__(self):
        super(DiceTestingScore, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        total_loss = 0

        for i in range(inputs.size(0)):
            input_i = inputs[i].view(-1)
            target_i = targets[i].view(-1)

            intersection = (input_i * target_i).sum()
            dice = (2. * intersection + smooth) / (input_i.sum() + target_i.sum() + smooth)

            total_loss += 1 - dice

        return total_loss
