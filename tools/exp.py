
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay





# class Focal_Loss(nn.Layer):
#     def __init__(self, class_num, alpha=0.25, gamma=2.0):
#         self.class_num = class_num
#         self.alpha = alpha
#         self.gamma = gamma

#         self.alpha = self.create_parameter([class_num, 1],
#                                             dtype='float32',
#                                             default_initializer=nn.initializer.Constant(value=alpha))
    
#     def forward(self, logit, label):
#         print(self.alpha)

def softmax_focal_loss(logit, label, class_num, alpha = 0.25, gamma = 2.0):
    """[summary]

    Args:
        logit ([type]): [description]
        label ([type]): [description]
        class_num ([type]): [description]
    """
    label_one_hot = F.one_hot(label, num_classes=class_num+1)
    label_one_hot.stop_gradient = True

    one = paddle.to_tensor([1.], dtype='float32')
    fg_label = paddle.greater_equal(label_one_hot, one)
    print(fg_label)
    fg_num = paddle.sum(paddle.cast(fg_label, dtype='float32'))
    print(fg_num)
    pred = F.softmax(logit)

    pt = (1 - pred) * label_one_hot + pred * (1 - label_one_hot)
    focal_weight = (alpha * label_one_hot + (1 - alpha) *(1 - label_one_hot)) * pt.pow(gamma)
    smooth_label = F.label_smooth(label_one_hot)
    loss = F.softmax_with_cross_entropy(pred, smooth_label, soft_label=True) * focal_weight

    return loss.sum()

pred = paddle.randn([1, 4])

label = paddle.to_tensor([0])


loss = softmax_focal_loss(pred, label, 3)

print(loss)
#print(focal_weight)