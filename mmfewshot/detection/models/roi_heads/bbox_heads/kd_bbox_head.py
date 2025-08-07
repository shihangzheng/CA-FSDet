import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.roi_heads.bbox_heads import ConvFCBBoxHead
from mmdet.models.losses import accuracy
from mmcv.ops.nms import batched_nms




@HEADS.register_module()
class IncreaseBBoxHead(ConvFCBBoxHead):

    def __init__(self,
                 fc_out_channels=1024,
                 scale=20.,
                 base_cpt=None,
                 base_alpha = 0.5,
                 *args,
                 **kwargs):
        super(IncreaseBBoxHead,
              self).__init__(*args,
                             **kwargs)
        # del self.fc_cls
        del self.shared_fcs
        del self.cls_fcs
        del self.reg_fcs
        assert base_cpt is not None
        self.base_cpt = base_cpt
        self.base_alpha = base_alpha

        num_base = self.num_classes // 4 * 3
        num_novel = self.num_classes // 4

        # base branch
        base_shared_fcs = nn.ModuleList()
        base_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        base_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.base_shared_fcs = base_shared_fcs
        # self.base_fc_cls = nn.Linear(self.cls_last_dim, num_base, bias=False)


        # novel branch
        novel_shared_fcs = nn.ModuleList()
        novel_shared_fcs.append(nn.Linear(49 * 256, fc_out_channels))
        novel_shared_fcs.append(nn.Linear(fc_out_channels, fc_out_channels))
        self.novel_shared_fcs = novel_shared_fcs

        if self.with_cls:
            self.fc_cls = nn.Linear(
                self.cls_last_dim, self.num_classes + 1, bias=False)

        # temperature
        self.scale = scale

        print(base_shared_fcs)


    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        torch.manual_seed(0)
        processing = True
        # During training, processing the checkpoints
        # During testing, directly load the checkpoints
        for param_name in state_dict:
            if 'novel_shared_fcs' in param_name:
                processing = False
                break
        print('-------processing-----', processing)

        if processing:
            # load base training checkpoint
            num_base_classes = self.num_classes // 4 * 3
            base_weights = torch.load(self.base_cpt, map_location='cpu')
            if 'state_dict' in base_weights:
                base_weights = base_weights['state_dict']

            for n, p in base_weights.items():
                if 'bbox_head' not in n:
                    state_dict[n] = copy.deepcopy(p)

            # initialize the base branch with the weight of base training
            for n, p in base_weights.items():
                if 'shared_fcs' in n:
                    new_n_base = n.replace('shared_fcs', 'base_shared_fcs')
                    state_dict[new_n_base] = copy.deepcopy(p)
                    new_n_novel = n.replace('shared_fcs', 'novel_shared_fcs')
                    state_dict[new_n_novel] = copy.deepcopy(p)

                if 'fc_cls' in n:
                    state_dict[n] = copy.deepcopy(p)

                if not self.reg_class_agnostic and 'fc_reg' in n:
                    state_dict[n] = copy.deepcopy(p)

            super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)


    def forward(self, x, return_fc_feat=False):

        x = x.flatten(1)

        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1-alpha) * novel_x
            x = self.relu(x)

        bbox_preds = self.fc_reg(x)
        
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist
        
        return scores, bbox_preds




@HEADS.register_module()
class KRBBoxHead(IncreaseBBoxHead):
    def __init__(self,
                 loss_kr_weight=0.001,
                 base_alpha = 0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)


        self.loss_kr = dict()
        self.loss_kr_weight = loss_kr_weight
        self.base_alpha = base_alpha

    def forward(self, x, return_fc_feat=False):
        

        loss_feature_kr = 0

        x = x.flatten(1)
        alpha = self.base_alpha

        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1-alpha) * novel_x
            if self.training:
                loss_feature_kr += torch.dist(alpha * base_x + (1-alpha) * novel_x, base_x, 2)

            x = self.relu(x)
        
        if self.training:

            loss_kr = loss_feature_kr / 2.0 * self.loss_kr_weight
            self.loss_kr['loss_kr'] = loss_kr


        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

    def loss(self,
             cls_score,
             bbox_pred,
             rois,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
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
       

        if  cls_score is not None:
            losses.update(self.loss_kr)

        return losses



@HEADS.register_module()
class NewKRBBoxHead(KRBBoxHead):

    def __init__(self,
                 MBD_DCALoss=None,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if MBD_DCALoss is not None:
            self.MBD_DCALoss = build_loss(copy.deepcopy(MBD_DCALoss))
        else:
            self.MBD_DCALoss = None



    def forward(self, x, return_fc_feat=False):
        
        kr_loss_list = []
        loss_feature_kr = 0
        x = x.flatten(1)
        alpha = self.base_alpha
        base_x = x
        assert len(self.base_shared_fcs) == len(self.novel_shared_fcs)
        for fc_ind in range(len(self.base_shared_fcs)):
            base_x = self.base_shared_fcs[fc_ind](x)
            novel_x = self.novel_shared_fcs[fc_ind](x)
            x = alpha * base_x + (1-alpha) * novel_x

            cos_sim = F.cosine_similarity(base_x, x, dim=-1)
            kr_loss_list.append(1 - cos_sim)
            x = self.relu(x.clone())
        kr_loss = torch.cat(kr_loss_list, dim=0)
        kr_loss = torch.mean(kr_loss)

        if self.training:
            kr_loss = kr_loss  * self.loss_kr_weight * 100
            self.loss_kr['loss_kr'] = kr_loss
 

        bbox_preds = self.fc_reg(x)
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 1e-5)
        with torch.no_grad():
            temp_norm = torch.norm(self.fc_cls.weight.data, p=2,
                               dim=1).unsqueeze(1).expand_as(
                                   self.fc_cls.weight.data)
            self.fc_cls.weight.data = self.fc_cls.weight.data.div(
                temp_norm + 1e-5)
        cos_dist = self.fc_cls(x_normalized)
        scores = self.scale * cos_dist

        return scores, bbox_preds

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
            
        if  cls_score is not None:
            losses.update(self.loss_kr)
            
        if self.MBD_DCALoss is not None and cls_score is not None:
            losses.update(self.MBD_DCALoss(cls_score, labels, label_weights))



        return losses



