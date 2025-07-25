# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${DATE} ${TIME}
# @Function   : function to 
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):    
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias, groups=groups)

def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=groups, stride=1)

def channel_shuffle(x, groups):
    b, c, h, w         = x.data.size()
    channels_per_group = c // groups
    x                  = x.view(b, groups, channels_per_group, h, w)
    x                  = torch.transpose(x, 1, 2).contiguous()
    x                  = x.view(b, -1, h, w)
    return x


class ShuffleUnit(nn.Module):
    def __init__(self, in_channels, out_channels, groups=3, grouped_conv=True, combine='add'):
        super(ShuffleUnit, self).__init__()

        self.in_channels         = in_channels
        self.out_channels        = out_channels
        self.grouped_conv        = grouped_conv
        self.combine             = combine
        self.groups              = groups
        self.bottleneck_channels = self.out_channels // 4

        if self.combine == 'add':
            self.depthwise_stride = 1
            self._combine_func    = self._add
        elif self.combine == 'concat':
            self.depthwise_stride = 2
            self._combine_func    = self._concat
            self.out_channels    -= self.in_channels
        else:
            raise ValueError("Cannot combine tensors with {} Only add and concat are supported".format(self.combine))

        self.first_1x1_groups    = self.groups if grouped_conv else 1
        self.g_conv_1x1_compress = self._make_grouped_conv1x1(
            self.in_channels,
            self.bottleneck_channels,
            self.first_1x1_groups,
            batch_norm=True,
            relu=True
            )

        self.depthwise_conv3x3  = conv3x3(self.bottleneck_channels, self.bottleneck_channels, stride=self.depthwise_stride, groups=self.bottleneck_channels)
        self.bn_after_depthwise = nn.BatchNorm2d(self.bottleneck_channels)
        self.g_conv_1x1_expand  = self._make_grouped_conv1x1(self.bottleneck_channels, self.out_channels, self.groups, batch_norm=True, relu=False)

    @staticmethod
    def _add(x, out):
        return x + out

    @staticmethod
    def _concat(x, out):
        return torch.cat((x, out), 1)

    def _make_grouped_conv1x1(self, in_channels, out_channels, groups, batch_norm=True, relu=False):
        modules            = OrderedDict()
        conv               = conv1x1(in_channels, out_channels, groups=groups)
        modules['conv1x1'] = conv

        if batch_norm:
            modules['batch_norm'] = nn.BatchNorm2d(out_channels)
        if relu:
            modules['relu']       = nn.ReLU()
        if len(modules) > 1:
            return nn.Sequential(modules)
        else:
            return conv


    def forward(self, x):
        residual = x

        if self.combine == 'concat':
            residual = F.avg_pool2d(residual, kernel_size=3, stride=2, padding=1)

        out = self.g_conv_1x1_compress(x)
        out = channel_shuffle(out, self.groups)
        out = self.depthwise_conv3x3(out)
        out = self.bn_after_depthwise(out)
        out = self.g_conv_1x1_expand(out)
        out = self._combine_func(residual, out)
        return F.relu(out)


class ShuffleNet(nn.Module):
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()
        self.groups        = groups
        self.stage_repeats = [3, 7, 3]
        self.in_channels   = in_channels
        self.num_classes   = num_classes

        if groups == 1:
            self.stage_out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            self.stage_out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            self.stage_out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            self.stage_out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            self.stage_out_channels = [-1, 24, 384, 768, 1536]
        else:
            raise ValueError("{} groups is not supported for 1x1 Grouped Convolutions".format(num_groups))
        self.conv1   = conv3x3(self.in_channels, self.stage_out_channels[1], stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stage2  = self._make_stage(2)
        self.stage3  = self._make_stage(3)
        self.stage4  = self._make_stage(4)
        num_inputs   = self.stage_out_channels[-1]
        self.fc      = nn.Linear(num_inputs, self.num_classes)


    def _make_stage(self, stage):
        modules      = OrderedDict()
        stage_name   = "ShuffleUnit_Stage{}".format(stage)
        grouped_conv = stage > 2 
        first_module = ShuffleUnit(
            self.stage_out_channels[stage-1],
            self.stage_out_channels[stage],
            groups       = self.groups,
            grouped_conv = grouped_conv,
            combine      = 'concat'
            )
        modules[stage_name+"_0"] = first_module

        for i in range(self.stage_repeats[stage-2]):
            name   = stage_name + "_{}".format(i+1)
            module = ShuffleUnit(
                self.stage_out_channels[stage],
                self.stage_out_channels[stage],
                groups       = self.groups,
                grouped_conv = True,
                combine      = 'add'
                )
            modules[name] = module

        return nn.Sequential(modules)


    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = F.avg_pool2d(x, x.data.size()[-2:])
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = ShuffleNet(in_channels=3, num_classes=1000, groups=8).to(device)
    x               = torch.randn(1, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(1, 3, 224, 224), device=device.type)
