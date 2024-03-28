import torch
import torch.nn as nn

from layers import CSPLayer, BaseConv


class YOLOXPAFPN(nn.Module):

    def __init__(self, in_channels=(128, 256, 512), act="silu"):

        super().__init__()
        self.in_channels = in_channels
        Conv = BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1, act=act)
        self.C3_p4 = CSPLayer(2 * in_channels[1], in_channels[1], 3, act=act)

        self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1, act=act)
        self.C3_p3 = CSPLayer(2 * in_channels[0], in_channels[0], 3, act=act)

        self.bu_conv2 = Conv(in_channels[0], in_channels[0], 3, 2, act=act)
        self.C3_n3 = CSPLayer(2 * in_channels[0], in_channels[1], 3, act=act)

        self.bu_conv1 = Conv(in_channels[1], in_channels[1], 3, 2, act=act)
        self.C3_n4 = CSPLayer(2 * in_channels[1], in_channels[2], 3, act=act)

    def forward(self, x):

        [x2, x1, x0] = x

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        f_out0 = self.C3_p4(f_out0)

        fpn_out1 = self.reduce_conv1(f_out0)
        f_out1 = self.upsample(fpn_out1)
        f_out1 = torch.cat([f_out1, x2], 1)
        pan_out2 = self.C3_p3(f_out1)

        p_out1 = self.bu_conv2(pan_out2)
        p_out1 = torch.cat([p_out1, fpn_out1], 1)
        pan_out1 = self.C3_n3(p_out1)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.C3_n4(p_out0)

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
