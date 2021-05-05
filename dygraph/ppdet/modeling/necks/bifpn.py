import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import XavierUniform
from paddle.regularizer import L2Decay
from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec


__all__ = ['BiFPN']

class SeparableConvBlock(nn.Layer):
    def __init__(self, in_channels, out_channels=None, 
                norm=True, norm_decay=0., norm_type='bn', norm_groups=32, freeze_norm=False, activation=False):
        super(SeparableConvBlock, self).__init__()
        self.norm_type = norm_type
        if out_channels is None:
            out_channels = in_channels

        # 不需要 bias
        self.depthwise_conv = nn.Conv2D(in_channels, in_channels, 
                                        kernel_size=3, stride=1, padding='SAME',
                                         groups=in_channels, bias_attr=False, 
                                         weight_attr=ParamAttr(initializer=XavierUniform(fan_out=in_channels)))
        # 需要 bias
        self.pointwise_conv = nn.Conv2D(in_channels, out_channels, 
                                        kernel_size=1, stride=1, padding='SAME',
                                        weight_attr=ParamAttr(initializer=XavierUniform(fan_out=in_channels)))

        self.norm = norm
        if self.norm:
            norm_lr = 0. if freeze_norm else 1.
            param_attr = ParamAttr(
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay),
                initializer=nn.initializer.Constant(value=1.0))
            bias_attr = ParamAttr(
                learning_rate=norm_lr,
                regularizer=L2Decay(norm_decay))
            
            if self.norm_type == 'bn':
                self.normlayer = nn.BatchNorm2D(num_features=out_channels, momentum=0.997, epsilon=1e-4,
                                        weight_attr=param_attr, bias_attr=bias_attr)
            elif self.norm_type == 'gn':
                self.normlayer = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels,
                                        weight_attr=param_attr, bias_attr=bias_attr)
        
        self.activation = activation
        if self.activation:
            self.swish = nn.Swish()

    
    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.normlayer(x)

        if self.activation:
            x = self.swish(x)

        return x

class ConvBNLayer(nn.Layer):

    def __init__(self, in_channels, out_channels, norm_decay=0., 
                    norm_type='bn', norm_groups=32, freeze_norm=False):
        super(ConvBNLayer, self).__init__()
        self.norm_type = norm_type

        norm_lr = 0. if freeze_norm else 1.
        param_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))
        bias_attr = ParamAttr(
            learning_rate=norm_lr,
            regularizer=L2Decay(norm_decay))

        if self.norm_type == 'bn':
            self.normlayer = nn.BatchNorm2D(num_features=out_channels, momentum=0.997, epsilon=1e-4,
                                    weight_attr=param_attr, bias_attr=bias_attr)
        elif self.norm_type == 'gn':
            self.normlayer = nn.GroupNorm(num_groups=norm_groups, num_channels=out_channels,
                                    weight_attr=param_attr, bias_attr=bias_attr)

        self.layer = nn.Sequential(
            nn.Conv2D(in_channels=in_channels, out_channels=out_channels, 
                    kernel_size=1, stride=1, padding='SAME'),
            self.normlayer,
        )
    
    def forward(self, x):
        return self.layer(x)


# @register
# @serializable
class BiFPNCell(nn.Layer):
    '''
        BiFPN 修改为适合ResNet结构输入

          P6_0 -------------------------> P6_2 -------->
            |-------------|                ↑
                            ↓                |
    C5    P5_0 ---------> P5_1 ---------> P5_2 -------->
            |-------------|--------------↑ ↑
                            ↓                |
    C4    P4_0 ---------> P4_1 ---------> P4_2 -------->
            |-------------|--------------↑ ↑
                            ↓                |
    C3    P3_0 ---------> P3_1 ---------> P3_2 -------->
            |-------------|--------------↑ ↑
                            |--------------↓ |
    C2    P2_0 -------------------------> P3_2 -------->
    '''
    def __init__(self, in_channels, out_channels, 
                first_time=True, freeze_norm=False, norm_decay=0.,
                norm_type='bn', norm_groups=32, epsilon=1e-4, attention=True):
        """
        Args:
            in_channels: 主干各个stage输入的通道
            out_channels: 统一到256
            first_time: whether the input comes directly from the efficientnet,
                        if True, downchannel it first, and downsample P5 to generate P6 then P7
            epsilon: epsilon of fast weighted attention sum of BiFPN, not the BN's epsilon
        """
        super(BiFPNCell, self).__init__()
        self.epsilon = epsilon
        self.first_time = first_time
        self.freeze_norm = freeze_norm
        self.out_channels = out_channels

        # Conv layers
        self.conv5_up = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv4_up = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv3_up = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv2_up = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)

        self.conv3_down = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv4_down = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv5_down = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)
        self.conv6_down = SeparableConvBlock(out_channels, norm_type=norm_type, norm_decay=norm_decay, freeze_norm=freeze_norm)

        # Feature scaling layers
        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p3_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)
        self.p4_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)
        self.p5_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)
        self.p6_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)

        self.swish = nn.Swish()

        if self.first_time:
            self.p5_down_channel = ConvBNLayer(in_channels[3], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)
            self.p4_down_channel = ConvBNLayer(in_channels[2], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)
            self.p3_down_channel = ConvBNLayer(in_channels[1], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)
            self.p2_down_channel = ConvBNLayer(in_channels[0], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)

            self.p5_to_p6 = nn.Sequential(
                ConvBNLayer(in_channels[3], out_channels, freeze_norm=self.freeze_norm),
                nn.MaxPool2D(kernel_size=3, stride=2, padding='SAME', ceil_mode=True)
            )

            self.p4_down_channel_2 = ConvBNLayer(in_channels[2], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)
            self.p5_down_channel_2 = ConvBNLayer(in_channels[3], out_channels, norm_decay=norm_decay, norm_type=norm_type, freeze_norm=freeze_norm)

        # Weight
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(value=1.0), trainable=True)
        self.p5_w1 = self.create_parameter(shape=[2], attr=weight_attr, dtype="float32")
        self.p5_w1_relu = nn.ReLU()
        self.p4_w1 = self.create_parameter(shape=[2], attr=weight_attr, dtype="float32")
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = self.create_parameter(shape=[2], attr=weight_attr, dtype="float32")
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = self.create_parameter(shape=[2], attr=weight_attr, dtype="float32")
        self.p2_w1_relu = nn.ReLU()

        self.p3_w2 = self.create_parameter(shape=[3], attr=weight_attr, dtype="float32")
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = self.create_parameter(shape=[3], attr=weight_attr, dtype="float32")
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = self.create_parameter(shape=[3], attr=weight_attr, dtype="float32")
        self.p5_w2_relu = nn.ReLU()
        self.p6_w2 = self.create_parameter(shape=[2], attr=weight_attr, dtype="float32")
        self.p6_w2_relu = nn.ReLU()

        self.attention = attention

    def forward(self, inputs):
        '''
            BiFPN 修改为适合ResNet结构输入

              P6_0 -------------------------> P6_2 -------->
                |-------------|                ↑
                                ↓                |
        C5    P5_0 ---------> P5_1 ---------> P5_2 -------->
                |-------------|--------------↑ ↑
                                ↓                |
        C4    P4_0 ---------> P4_1 ---------> P4_2 -------->
                |-------------|--------------↑ ↑
                                ↓                |
        C3    P3_0 ---------> P3_1 ---------> P3_2 -------->
                |-------------|--------------↑ ↑
                                |--------------↓ |
        C2    P2_0 -------------------------> P2_2 -------->
        '''
        if self.attention:
            return self._forward_fast_attention(inputs)
        else: # TODO
            pass

    def _forward_fast_attention(self, inputs):
        if self.first_time:
            p2, p3, p4, p5 = inputs
            # 继续卷积一次
            p6_in = self.p5_to_p6(p5)
            # 调整通道
            p2_in = self.p2_down_channel(p2)
            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)
        else:
            p2_in, p3_in, p4_in, p5_in, p6_in = inputs
        
        # 上采样 向下 

        # Weights : P6_0 + P5_0 --> P5_1
        p5_w1 = self.p5_w1_relu(self.p5_w1)
        weight = p5_w1 / (paddle.sum(p5_w1) + self.epsilon)
        # Connection : P6_0 + P5_0 --> P5_1
        # BUG fix: 指定形状大小
        p6_temp = F.interpolate(p6_in, size=[p5_in.shape[2], p5_in.shape[3]])
        p5_up = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * p6_temp))

        # Weights : P5_1 + P4_0 --> P4_1
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (paddle.sum(p4_w1) + self.epsilon)
        # Connection : P5_1 + P4_0 --> P4_1
        p4_up = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p5_upsample(p5_up)))

        # Weights : P4_1 + P3_0 --> P3_1
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (paddle.sum(p3_w1) + self.epsilon)
        # Connection : P4_1 + P3_0 --> P3_1
        p3_up = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p4_upsample(p4_up)))

        # Weights : P3_1 + P2_0 --> P2_2
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (paddle.sum(p2_w1) + self.epsilon)
        # Connection : P3_1 + P2_0 --> P2_2
        p2_out = self.conv2_up(self.swish(weight[0] * p2_in + weight[1] * self.p3_upsample(p3_up)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # 下采样 向上 + skip

        # Weights : (w0*P3_1 + w1*P2_2) + w2*P3_0 --> P3_2
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (paddle.sum(p3_w2) + self.epsilon)
        # Connection : (w0*P3_1 + w1*P2_2) + w2*P3_0 --> P3_2
        p3_out = self.conv3_down(
            self.swish(weight[0] * p3_in + weight[1] * p3_up + weight[2] * self.p3_downsample(p2_out)))

        # Weights : (w0*P4_1 + w1*P3_2) + w2*P4_0 --> P4_2
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (paddle.sum(p4_w2) + self.epsilon)
        # Connection : (w0*P4_1 + w1*P3_2) + w2*P4_0 --> P4_2
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_up + weight[2] * self.p4_downsample(p3_out)))

        # Weights : (w0*P5_1 + w1*P4_2) + w2*P4_0 --> P5_2
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (paddle.sum(p5_w2) + self.epsilon)
        # Connection : (w0*P5_1 + w1*P4_2) + w2*P4_0 --> P5_2
        p5_out = self.conv5_down(
            self.swish(weight[0] * p5_in + weight[1] * p5_up + weight[2] * self.p5_downsample(p4_out)))

        # Weights : P5_2 + P6_0 --> P6_2
        p6_w2 = self.p6_w2_relu(self.p6_w2)
        weight = p6_w2 / (paddle.sum(p6_w2) + self.epsilon)
        # Connection : P5_2 + P6_0 --> P6_2
        p6_out = self.conv6_down(self.swish(weight[0] * p6_in + weight[1] * self.p6_downsample(p5_out)))

        return p2_out, p3_out, p4_out, p5_out, p6_out


@register
@serializable
class BiFPN(nn.Layer):
    def __init__(self, in_channels, out_channels, 
                num_cell=1, freeze_norm=False, norm_type='bn', norm_decay=0.,
                norm_groups=32, epsilon=1e-4, attention=True,
                spatial_scales=[1. / 4, 1. / 8, 1. / 16, 1. / 32, 1. / 64]):
        super(BiFPN, self).__init__()
        assert norm_type in ['bn', 'gn']
        assert num_cell >= 1 & isinstance(num_cell, int)
        self.norm_type = norm_type
        self.spatial_scales = spatial_scales
        self.out_channels = out_channels
        self.num_cell = num_cell

        self.cell_1 = BiFPNCell(in_channels, out_channels, first_time=True, 
                                freeze_norm=freeze_norm, norm_decay=norm_decay, norm_type=norm_type,
                                epsilon=epsilon, attention=attention)

        if self.num_cell > 1:
            self.cells = nn.LayerList(
                [BiFPNCell([256, 256, 256, 256], out_channels, first_time=False, 
                            freeze_norm=freeze_norm, norm_decay=norm_decay, norm_type=norm_type,
                            epsilon=epsilon, attention=attention) for _ in range(self.num_cell-1)]
            )

    def forward(self, feats):
        out = self.cell_1(feats)
        if self.num_cell > 1:
            for cell in self.cells:
                out = cell(out)
        
        return out

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channels, stride=1. / s)
            for s in self.spatial_scales
        ]