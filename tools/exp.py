
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay

class SeparableConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels=None, norm=True, norm_decay=0., freeze_norm=False, activation=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # 不需要 bias
        self.depthwise_conv = nn.Conv2D(in_channels, in_channels, 
                                        kernel_size=3, stride=1, padding='same',
                                         groups=in_channels, bias_attr=False)
        # 需要 bias
        self.pointwise_conv = nn.Conv2D(in_channels, out_channels, 
                                        kernel_size=1, stride=1, padding='same')

        self.norm = norm
        if self.norm:
            norm_lr = 0. if freeze_norm else 1.
            param_attr = ParamAttr(
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))
            bias_attr = ParamAttr(
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))
            self.bn = nn.BatchNorm2D(num_features=out_channels, momentum=0.99, epsilon=1e-3,
                                    weight_attr=param_attr, bias_attr=bias_attr)
        
        self.activation = activation
        if self.activation:
            self.swish = nn.Swish()

    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)

        if self.activation:
            x = self.swish(x)

        return x

class ConvBNLayer(nn.Layer):

    def __init__(self, in_channels, out_channels, norm_decay=0., freeze_norm=False):
        super(ConvBNLayer, self).__init__()

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        self.layer = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2D(out_channels, momentum=0.99, epsilon=1e-3,
            weight_attr=param_attr, bias_attr=bias_attr),
        )
    
    def forward(self, x):
        return self.layer(x)



c5 = paddle.randn([2, 2048, 23, 41])

conv = nn.Sequential(
    ConvBNLayer(2048, 256, freeze_norm=False),
    nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)
)

out = conv(c5)
print(out.shape)
out = F.interpolate(out, size=[23, 41])
print(out.shape)