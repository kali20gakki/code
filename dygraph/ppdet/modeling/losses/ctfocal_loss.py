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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable

__all__ = ['CTFocalLoss', 'SoftmaxFocalLoss', 'SigmoidFocalLoss']


@register
@serializable
class CTFocalLoss(object):
    """
    CTFocalLoss
    Args:
        loss_weight (float):  loss weight
        gamma (float):  gamma parameter for Focal Loss
    """

    def __init__(self, loss_weight=1., gamma=2.0):
        self.loss_weight = loss_weight
        self.gamma = gamma

    def __call__(self, pred, target):
        """
        Calculate the loss
        Args:
            pred(Tensor): heatmap prediction
            target(Tensor): target for positive samples
        Return:
            ct_focal_loss (Tensor): Focal Loss used in CornerNet & CenterNet.
                Note that the values in target are in [0, 1] since gaussian is
                used to reduce the punishment and we treat [0, 1) as neg example.
        """
        fg_map = paddle.cast(target == 1, 'float32')
        fg_map.stop_gradient = True
        bg_map = paddle.cast(target < 1, 'float32')
        bg_map.stop_gradient = True

        neg_weights = paddle.pow(1 - target, 4) * bg_map
        pos_loss = 0 - paddle.log(pred) * paddle.pow(1 - pred,
                                                     self.gamma) * fg_map
        neg_loss = 0 - paddle.log(1 - pred) * paddle.pow(
            pred, self.gamma) * neg_weights
        pos_loss = paddle.sum(pos_loss)
        neg_loss = paddle.sum(neg_loss)

        fg_num = paddle.sum(fg_map)
        ct_focal_loss = (pos_loss + neg_loss) / (
            fg_num + paddle.cast(fg_num == 0, 'float32'))
        return ct_focal_loss * self.loss_weight


def softmax_focal_loss(logit, label, class_num, alpha = 0.25, gamma = 2.0, reduction='mean'):
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


@register
@serializable
class SoftmaxFocalLoss(object):

    def __init__(self, loss_weight=1., alpha = 0.25, gamma = 2.0, reduction='mean'):
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logit, label, class_num):
        label_one_hot = F.one_hot(label, num_classes=class_num+1)
        label_one_hot.stop_gradient = True

        pred = F.softmax(logit)
        pt = (1 - pred) * label_one_hot + pred * (1 - label_one_hot)
        focal_weight = (self.alpha * label_one_hot + (1 - self.alpha) *(1 - label_one_hot)) * pt.pow(self.gamma)
        smooth_label = F.label_smooth(label_one_hot)
        loss = F.softmax_with_cross_entropy(pred, smooth_label, soft_label=True) * focal_weight

        if self.reduction == 'sum':
            loss = paddle.sum(loss)
        elif self.reduction == 'mean':
            loss = paddle.mean(loss)

        return loss


@register
@serializable
class SigmoidFocalLoss(object):

    def __init__(self, loss_weight=1., alpha = 0.25, gamma = 2.0, reduction='mean'):
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logit, label, class_num):
        label_one_hot = F.one_hot(label, num_classes=class_num+1)
        label_one_hot.stop_gradient = True

        loss = F.sigmoid_focal_loss(logit, label_one_hot, 
                                    alpha=self.alpha, 
                                    gamma=self.gamma,
                                    normalizer=label_one_hot[0],
                                    reduction=self.reduction)

        return loss