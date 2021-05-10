from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn.functional as F
from ppdet.core.workspace import register, serializable
from typing import Union

__all__ = ['SeesawLoss']

@register
@serializable
class SeesawLoss(object):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    p: Parameter for Mitigation Factor,
       Set to 0.8 for default following the paper.
    q: Parameter for Compensation Factor
       Set to 2 for default following the paper.
    class_num: Class nums
    """
    def __init__(self, p: float = 0.8, q: float = 2):
        super().__init__()
        self.eps = 1.0e-6
        self.p = p
        self.q = q
        self.class_counts = None

    def __call__(self, logit, label, class_num):
        label_one_hot = F.one_hot(label, class_num+1) # [512, 4]
        #print(label)

        # Mitigation Factor 
        # 在线统计类别数
        if self.class_counts is None:
            self.class_counts = (label_one_hot.sum(axis=0) + 1).astype('float32') # to prevent devided by zero.
        else:
            self.class_counts += label_one_hot.sum(axis=0)

        m_conditions = self.class_counts.reshape([-1, 1]) > self.class_counts.reshape([1, -1])
        m_trues = (self.class_counts.reshape([1, -1]) / self.class_counts.reshape([-1, 1])) ** self.p
        m_falses = paddle.ones([len(self.class_counts), len(self.class_counts)])
        m = paddle.where(m_conditions, m_trues, m_falses)   # [num_labels, num_labels]

        # Compensation Factor
        # only error sample need to compute Compensation Factor
        probility = F.softmax(logit, axis=-1)
        c_condition = probility / paddle.unsqueeze(paddle.sum((probility * label_one_hot), axis=-1), -1)
        c_condition = paddle.stack([c_condition] * label_one_hot.shape[-1], axis=1) # [B, N, N]
        c_condition = c_condition * paddle.unsqueeze(label_one_hot, -1)  # [B, N, N]
        false = paddle.ones(c_condition.shape)
        c = paddle.where(c_condition>1, c_condition ** self.q, false) # [B, N, N]

        # Sij = Mij * Cij 
        s = paddle.unsqueeze(m, 0) * c
        # softmax trick to prevent overflow (like logsumexp trick)
        max_element = logit.max(axis=-1)

        logit = logit - paddle.unsqueeze(max_element, -1)  # to prevent overflow
        numerator = paddle.exp(logit)
        denominator = (
            paddle.unsqueeze((1 - label_one_hot), 1)
            * paddle.unsqueeze(s, 0)
            * paddle.unsqueeze(paddle.exp(logit), 1)).sum(axis=-1) \
            + paddle.exp(logit)

        sigma = numerator / (denominator + self.eps)
        loss = (- label_one_hot * paddle.log(sigma + self.eps)).sum(axis=-1)
        return loss.mean()


# x = paddle.to_tensor([3,2,1,0])
# m_conditions = x.reshape([-1, 1]) > x.reshape([1, -1])
# print(m_conditions)
# m_trues = (x.reshape([1, -1]) / x.reshape([-1, 1])) ** 0.8
# print(m_trues)
# m_falses = paddle.ones([len(x), len(x)])
# print(m_falses)
# m = paddle.where(m_conditions, m_trues, m_falses)
# print(m)