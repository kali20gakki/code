# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

import paddle
import paddle.nn as nn
import numpy as np
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, XavierUniform, KaimingNormal
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, create
from ppdet.modeling import ops

from .roi_extractor import RoIAlign
from ..shape_spec import ShapeSpec
from ..bbox_utils import bbox2delta, delta2bbox
from ppdet.modeling.layers import ConvNormLayer
from ..losses.iou_loss import *
__all__ = ['TwoFCHead', 'XConvNormHead', 'BBoxHead']


@register
class TwoFCHead(nn.Layer):
    def __init__(self, in_dim=256, mlp_dim=1024, resolution=7):
        super(TwoFCHead, self).__init__()
        self.in_dim = in_dim
        self.mlp_dim = mlp_dim
        fan = in_dim * resolution * resolution
        self.fc6 = nn.Linear(
            in_dim * resolution * resolution,
            mlp_dim,
            weight_attr=paddle.ParamAttr(
                initializer=XavierUniform(fan_out=fan)))

        self.fc7 = nn.Linear(
            mlp_dim,
            mlp_dim,
            weight_attr=paddle.ParamAttr(initializer=XavierUniform()))

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_dim': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.mlp_dim, )]

    def forward(self, rois_feat):
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = self.fc6(rois_feat)
        fc6 = F.relu(fc6)
        fc7 = self.fc7(fc6)
        fc7 = F.relu(fc7)
        return fc7


@register
class XConvNormHead(nn.Layer):
    """
    RCNN bbox head with serveral convolution layers
    Args:
        in_dim(int): num of channels for the input rois_feat
        num_convs(int): num of convolution layers for the rcnn bbox head
        conv_dim(int): num of channels for the conv layers
        mlp_dim(int): num of channels for the fc layers
        resolution(int): resolution of the rois_feat
        norm_type(str): norm type, 'gn' by defalut
        freeze_norm(bool): whether to freeze the norm
        stage_name(str): used in CascadeXConvNormHead, '' by default
    """
    __shared__ = ['norm_type', 'freeze_norm']

    def __init__(self,
                 in_dim=256,
                 num_convs=4,
                 conv_dim=256,
                 mlp_dim=1024,
                 resolution=7,
                 norm_type='gn',
                 freeze_norm=False,
                 stage_name=''):
        super(XConvNormHead, self).__init__()
        self.in_dim = in_dim
        self.num_convs = num_convs
        self.conv_dim = conv_dim
        self.mlp_dim = mlp_dim
        self.norm_type = norm_type
        self.freeze_norm = freeze_norm

        self.bbox_head_convs = []
        fan = conv_dim * 3 * 3
        initializer = KaimingNormal(fan_in=fan)
        for i in range(self.num_convs):
            in_c = in_dim if i == 0 else conv_dim
            head_conv_name = stage_name + 'bbox_head_conv{}'.format(i)
            head_conv = self.add_sublayer(
                head_conv_name,
                ConvNormLayer(
                    ch_in=in_c,
                    ch_out=conv_dim,
                    filter_size=3,
                    stride=1,
                    norm_type=self.norm_type,
                    norm_name=head_conv_name + '_norm',
                    freeze_norm=self.freeze_norm,
                    initializer=initializer,
                    name=head_conv_name))
            self.bbox_head_convs.append(head_conv)

        fan = conv_dim * resolution * resolution
        self.fc6 = nn.Linear(
            conv_dim * resolution * resolution,
            mlp_dim,
            weight_attr=paddle.ParamAttr(
                initializer=XavierUniform(fan_out=fan)),
            bias_attr=paddle.ParamAttr(
                learning_rate=2., regularizer=L2Decay(0.)))

    @classmethod
    def from_config(cls, cfg, input_shape):
        s = input_shape
        s = s[0] if isinstance(s, (list, tuple)) else s
        return {'in_dim': s.channels}

    @property
    def out_shape(self):
        return [ShapeSpec(channels=self.mlp_dim, )]

    def forward(self, rois_feat):
        for i in range(self.num_convs):
            rois_feat = F.relu(self.bbox_head_convs[i](rois_feat))
        rois_feat = paddle.flatten(rois_feat, start_axis=1, stop_axis=-1)
        fc6 = F.relu(self.fc6(rois_feat))
        return fc6

def softmax_focal_loss(logit, label, class_num, alpha = 0.25, gamma = 2.0, reduction='sum'):
    """[summary]

    Args:
        logit ([type]): [description]
        label ([type]): [description]
        class_num ([type]): [description]
    """
    label_one_hot = F.one_hot(label, num_classes=class_num+1)
    label_one_hot.stop_gradient = True

    # one = paddle.to_tensor([1.], dtype='float32')
    # fg_label = paddle.greater_equal(label_one_hot, one)
    # fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
    pred = F.softmax(logit)

    pt = (1 - pred) * label_one_hot + pred * (1 - label_one_hot)
    focal_weight = (alpha * label_one_hot + (1 - alpha) *(1 - label_one_hot)) * pt.pow(gamma)
    smooth_label = F.label_smooth(label_one_hot)
    loss = F.softmax_with_cross_entropy(pred, smooth_label, soft_label=True) * focal_weight

    if reduction == 'sum':
        loss = paddle.sum(loss)
    elif reduction == 'mean':
        loss = paddle.mean(loss)

    return loss


def diou_loss(pbox, gbox, iou_weight=1., eps=1e-10, use_complete_iou_loss=True):
    x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
    x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1

    cxg = (x1g + x2g) / 2
    cyg = (y1g + y2g) / 2
    wg = x2g - x1g
    hg = y2g - y1g

    x2 = paddle.maximum(x1, x2)
    y2 = paddle.maximum(y1, y2)

    # A and B
    xkis1 = paddle.maximum(x1, x1g)
    ykis1 = paddle.maximum(y1, y1g)
    xkis2 = paddle.minimum(x2, x2g)
    ykis2 = paddle.minimum(y2, y2g)

    # A or B
    xc1 = paddle.minimum(x1, x1g)
    yc1 = paddle.minimum(y1, y1g)
    xc2 = paddle.maximum(x2, x2g)
    yc2 = paddle.maximum(y2, y2g)

    intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
    intsctk = intsctk * paddle.greater_than(
        xkis2, xkis1) * paddle.greater_than(ykis2, ykis1)
    unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                    ) - intsctk + eps
    iouk = intsctk / unionk

    # DIOU term
    dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
    dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
    diou_term = (dist_intersection + eps) / (dist_union + eps)

    # CIOU term
    ciou_term = 0
    if use_complete_iou_loss:
        ar_gt = wg / hg
        ar_pred = w / h
        arctan = paddle.atan(ar_gt) - paddle.atan(ar_pred)
        ar_loss = 4. / np.pi / np.pi * arctan * arctan
        alpha = ar_loss / (1 - iouk + ar_loss + eps)
        alpha.stop_gradient = True
        ciou_term = alpha * ar_loss

    diou = paddle.mean((1 - iouk + ciou_term + diou_term) * iou_weight)

    return diou


@register
class BBoxHead(nn.Layer):
    __shared__ = ['num_classes']
    __inject__ = ['bbox_assigner']
    """
    head (nn.Layer): Extract feature in bbox head
    in_channel (int): Input channel after RoI extractor
    roi_extractor (object): The module of RoI Extractor
    bbox_assigner (object): The module of Box Assigner, label and sample the 
                            box.
    with_pool (bool): Whether to use pooling for the RoI feature.
    num_classes (int): The number of classes
    bbox_weight (List[float]): The weight to get the decode box 
    """

    def __init__(self,
                 head,
                 in_channel,
                 roi_extractor=RoIAlign().__dict__,
                 bbox_assigner='BboxAssigner',
                 with_pool=False,
                 num_classes=80,
                 bbox_weight=[10., 10., 5., 5.]):
        super(BBoxHead, self).__init__()
        self.head = head
        self.roi_extractor = roi_extractor
        if isinstance(roi_extractor, dict):
            self.roi_extractor = RoIAlign(**roi_extractor)
        self.bbox_assigner = bbox_assigner

        self.with_pool = with_pool
        self.num_classes = num_classes
        self.bbox_weight = bbox_weight

        lr_factor = 1.
        self.bbox_score = nn.Linear(
            in_channel,
            self.num_classes + 1,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.01)))

        self.bbox_delta = nn.Linear(
            in_channel,
            4 * self.num_classes,
            weight_attr=paddle.ParamAttr(initializer=Normal(
                mean=0.0, std=0.001)))
        self.assigned_label = None
        self.assigned_rois = None

    @classmethod
    def from_config(cls, cfg, input_shape):
        roi_pooler = cfg['roi_extractor']
        assert isinstance(roi_pooler, dict)
        kwargs = RoIAlign.from_config(cfg, input_shape)
        roi_pooler.update(kwargs)
        kwargs = {'input_shape': input_shape}
        head = create(cfg['head'], **kwargs)
        return {
            'roi_extractor': roi_pooler,
            'head': head,
            'in_channel': head.out_shape[0].channels
        }

    def forward(self, body_feats=None, rois=None, rois_num=None, inputs=None):
        """
        body_feats (list[Tensor]): Feature maps from backbone
        rois (list[Tensor]): RoIs generated from RPN module
        rois_num (Tensor): The number of RoIs in each image
        inputs (dict{Tensor}): The ground-truth of image
        """
        if self.training:
            rois, rois_num, targets = self.bbox_assigner(rois, rois_num, inputs)
            self.assigned_rois = (rois, rois_num)
            self.assigned_targets = targets

        rois_feat = self.roi_extractor(body_feats, rois, rois_num)
        bbox_feat = self.head(rois_feat)
        if self.with_pool:
            feat = F.adaptive_avg_pool2d(bbox_feat, output_size=1)
            feat = paddle.squeeze(feat, axis=[2, 3])
        else:
            feat = bbox_feat
        scores = self.bbox_score(feat)
        deltas = self.bbox_delta(feat)

        if self.training:
            loss = self.get_loss(scores, deltas, targets, rois,
                                 self.bbox_weight)
            return loss, bbox_feat
        else:
            pred = self.get_prediction(scores, deltas)
            return pred, self.head

    def get_loss(self, scores, deltas, targets, rois, bbox_weight):
        """
        scores (Tensor): scores from bbox head outputs
        deltas (Tensor): deltas from bbox head outputs
        targets (list[List[Tensor]]): bbox targets containing tgt_labels, tgt_bboxes and tgt_gt_inds
        rois (List[Tensor]): RoIs generated in each batch
        """
        # TODO: better pass args
        tgt_labels, tgt_bboxes, tgt_gt_inds = targets
        tgt_labels = paddle.concat(tgt_labels) if len(
            tgt_labels) > 1 else tgt_labels[0]
        tgt_labels = tgt_labels.cast('int64')
        tgt_labels.stop_gradient = True
        
        # CE -> Focal loss
        # loss_bbox_cls = F.cross_entropy(
        #     input=scores, label=tgt_labels, reduction='mean')
        loss_bbox_cls = softmax_focal_loss(scores, tgt_labels, self.num_classes)

        # bbox reg
        cls_agnostic_bbox_reg = deltas.shape[1] == 4

        fg_inds = paddle.nonzero(
            paddle.logical_and(tgt_labels >= 0, tgt_labels <
                               self.num_classes)).flatten()

        cls_name = 'loss_bbox_cls'
        reg_name = 'loss_bbox_reg'
        loss_bbox = {}

        loss_weight = 1.
        if fg_inds.numel() == 0:
            fg_inds = paddle.zeros([1], dtype='int32')
            loss_weight = 0.

        if cls_agnostic_bbox_reg:
            reg_delta = paddle.gather(deltas, fg_inds)
        else:
            fg_gt_classes = paddle.gather(tgt_labels, fg_inds)

            reg_row_inds = paddle.arange(fg_gt_classes.shape[0]).unsqueeze(1)
            reg_row_inds = paddle.tile(reg_row_inds, [1, 4]).reshape([-1, 1])

            reg_col_inds = 4 * fg_gt_classes.unsqueeze(1) + paddle.arange(4)

            reg_col_inds = reg_col_inds.reshape([-1, 1])
            reg_inds = paddle.concat([reg_row_inds, reg_col_inds], axis=1)

            reg_delta = paddle.gather(deltas, fg_inds)
            reg_delta = paddle.gather_nd(reg_delta, reg_inds).reshape([-1, 4])

        rois = paddle.concat(rois) if len(rois) > 1 else rois[0]
        #reg_bboxes = delta2bbox(deltas, rois, bbox_weight)
        tgt_bboxes = paddle.concat(tgt_bboxes) if len(
            tgt_bboxes) > 1 else tgt_bboxes[0]

        reg_target = bbox2delta(rois, tgt_bboxes, bbox_weight)
        reg_target = paddle.gather(reg_target, fg_inds)
        reg_target.stop_gradient = True
        loss_bbox_reg = paddle.abs(reg_delta - reg_target).sum(
        ) / tgt_labels.shape[0]


        loss_bbox[cls_name] = loss_bbox_cls * loss_weight
        loss_bbox[reg_name] = loss_bbox_reg * loss_weight

        return loss_bbox

    def get_prediction(self, score, delta):
        bbox_prob = F.softmax(score)
        return delta, bbox_prob

    def get_head(self, ):
        return self.head

    def get_assigned_targets(self, ):
        return self.assigned_targets

    def get_assigned_rois(self, ):
        return self.assigned_rois
