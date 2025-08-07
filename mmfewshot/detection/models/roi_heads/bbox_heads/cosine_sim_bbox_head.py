# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from torch import Tensor


@HEADS.register_module()
class CosineSimBBoxHead(ConvFCBBoxHead):

    """BBOxHead for `TFA <https://arxiv.org/abs/2003.06957>`_.

    The code is modified from the official implementation
    https://github.com/ucbdrive/few-shot-object-detection/

    Args:
        scale (int): Scaling factor of `cls_score`. Default: 20.   # `scale` 参数定义了分类分数的缩放因子，默认值为 20。 这个缩放因子可以调整分类分数的大小，从而平衡不同类别间的分类精度。
        learnable_scale (bool): Learnable global scaling factor.   # `learnable_scale` 参数决定缩放因子是否可学习。默认值是 False。 如果设为 True，模型可以在训练过程中动态调整分类分数的缩放值。
            Default: False.
        eps (float): Constant variable to avoid division by zero.  # `eps` 参数是一个很小的常数，用来避免在计算余弦相似度时除零错误。 当两个向量的模为零时，直接相除会导致错误，加入 eps 可以保证计算稳定性。
    """

    def __init__(self,
                 scale: int = 20,
                 learnable_scale: bool = False,
                 eps: float = 1e-5,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # learnable global scaling factor
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1) * scale)
        else:
            self.scale = scale
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward function.  前向传播函数，用于计算分类分数和边界框回归结果。

        Args:
            x (Tensor): Shape of (num_proposals, C, H, W).  x (Tensor): 输入的张量，形状为 (num_proposals, C, H, W)，
        其中 `num_proposals` 是建议框的数量，C 是通道数，H 和 W 分别是特征图的高度和宽度。

        Returns:
            tuple:
                cls_score (Tensor): Cls scores, has shape   cls_score (Tensor): 分类分数，形状为 (num_proposals, num_classes)。
                    (num_proposals, num_classes).
                bbox_pred (Tensor): Box energies / deltas, has shape   bbox_pred (Tensor): 边界框回归结果，表示回归的四个坐标（dx, dy, dw, dh），
                    (num_proposals, 4).    形状为 (num_proposals, 4)。
        """
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_cls = x
        x_reg = x

        for conv in self.cls_convs:
            x_cls = conv(x_cls)
        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)
        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None

        if x_cls.dim() > 2:
            x_cls = torch.flatten(x_cls, start_dim=1)

        # normalize the input x along the `input_size` dimension
        x_norm = torch.norm(x_cls, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_cls_normalized = x_cls.div(x_norm + self.eps)
        # normalize weight
        with torch.no_grad():
            temp_norm = torch.norm(
                self.fc_cls.weight, p=2,
                dim=1).unsqueeze(1).expand_as(self.fc_cls.weight)
            self.fc_cls.weight.div_(temp_norm + self.eps)
        # calculate and scale cls_score
        cls_score = self.scale * self.fc_cls(
            x_cls_normalized) if self.with_cls else None

        return cls_score, bbox_pred



@HEADS.register_module()
class NewCosSimBBoxHead(CosineSimBBoxHead):
    def __init__(self,
                 MBD_DCALoss=None,
                 with_weight_decay=False,
                 decay_rate=1.0,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if MBD_DCALoss is not None:
            self.MBD_DCALoss = build_loss(copy.deepcopy(MBD_DCALoss))
        else:
            self.MBD_DCALoss = None


        self.with_weight_decay = with_weight_decay
        # This will be updated by :class:`ContrastiveLossDecayHook`
        # in the training phase.
        self._decay_rate = decay_rate

    def set_decay_rate(self, decay_rate: float):
        """Contrast loss weight decay hook will set the `decay_rate` according
        to iterations.

        Args:
            decay_rate (float): Decay rate for weight decay.
        """
        self._decay_rate = decay_rate

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             cos_dis=None,
             reduction_override=None):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['acc'] = accuracy(cls_score, labels)

        if bbox_pred is not None:
            bg_class_ind = self.num_classes
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            pos_inds = (labels >= 0) & (labels < bg_class_ind)
            # not use bbox sampling
            # pos_inds[1024:] = False
            # do not perform bounding box regression for BG anymore.
            if pos_inds.any():
                if self.reg_decoded_bbox:
                    # When the regression loss (e.g. `IouLoss`,
                    # `GIouLoss`, `DIouLoss`) is applied directly on
                    # the decoded bounding boxes, it decodes the
                    # already encoded coordinates to absolute format.
                    bbox_pred = self.bbox_coder.decode(rois[:, 1:], bbox_pred)
                if self.reg_class_agnostic:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), 4)[pos_inds.type(torch.bool)]
                else:
                    pos_bbox_pred = bbox_pred.view(
                        bbox_pred.size(0), -1,
                        4)[pos_inds.type(torch.bool),
                           labels[pos_inds.type(torch.bool)]]
                losses['loss_bbox'] = self.loss_bbox(
                    pos_bbox_pred,
                    bbox_targets[pos_inds.type(torch.bool)],
                    bbox_weights[pos_inds.type(torch.bool)],
                    avg_factor=bbox_targets.size(0),
                    reduction_override=reduction_override)
            else:
                losses['loss_bbox'] = bbox_pred[pos_inds].sum()


        if self.with_weight_decay:
            decay_rate = self._decay_rate
        else:
            decay_rate = None
            

        if self.MBD_DCALoss is not None and cls_score is not None:
            losses.update(self.MBD_DCALoss(cls_score, labels, label_weights, decay_rate))

        return losses