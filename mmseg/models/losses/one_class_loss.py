import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .dice_loss import binary_dice_loss
from .utils import weighted_loss, weight_reduce_loss, reduce_loss


@LOSSES.register_module()
class BCEDiceLoss(nn.Module):
    """DiceLoss.

    This loss is proposed in `V-Net: Fully Convolutional Neural Networks for
    Volumetric Medical Image Segmentation <https://arxiv.org/abs/1606.04797>`_.

    Args:
        loss_type (str, optional): Binary or multi-class loss.
            Default: 'multi_class'. Options are "binary" and "multi_class".
        smooth (float): A float number to smooth loss, and avoid NaN error.
            Default: 1
        exponent (float): An float number to calculate denominator
            value: \\sum{x^exponent} + \\sum{y^exponent}. Default: 2.
        reduction (str, optional): The method used to reduce the loss. Options
            are "none", "mean" and "sum". This parameter only works when
            per_image is True. Default: 'mean'.
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Default to 1.0.
        ignore_index (int | None): The label index to be ignored. Default: 255.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_dice'.
    """

    def __init__(self,
                 smooth=1,
                 exponent=2,
                 reduction='mean',
                 loss_weight=1.0,
                 loss_name='loss_bce_dice',
                 loss_multiply=(1.0, 3.0),
                 ignore_index=255):
        super(BCEDiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name
        self.bce_loss = F.binary_cross_entropy
        self.dice_loss = binary_dice_loss
        self.loss_multiply = loss_multiply
        self.ignore_index = ignore_index

    def forward(self,
                pred,
                target,
                **kwargs):

        pred = F.sigmoid(pred).squeeze(1)

        valid_mask = (target != self.ignore_index).long()

        bce = self.bce_loss(pred, target.float(), reduction='none')
        bce = weight_reduce_loss(bce, weight=valid_mask, reduction='mean')
        dice = self.dice_loss(pred, target, valid_mask=valid_mask, smooth=self.smooth, exponent=self.exponent)
        loss = self.loss_multiply[0] * bce + self.loss_multiply[1] * dice
        loss = self.loss_weight * loss

        return loss

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

