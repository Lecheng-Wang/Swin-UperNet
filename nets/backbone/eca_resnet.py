# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${2025/03/16} ${17:14}
# @Function: Realization of resnet architecture

import math
import torch
import torch.nn as nn


# ECANet Block after BasicBlock or BottleNeck
class ECANet_Block(nn.Module):
    def __init__(self, in_channels, gamma=2, b=1):
        super(ECANet_Block, self).__init__()
        kernel_size   = int(abs((math.log(in_channels, 2) + b) / gamma))
        kernel_size   = kernel_size if kernel_size % 2 else kernel_size + 1
        padding       = kernel_size//2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid  = nn.Sigmoid()

    def forward (self, x):
        B, C, H, W = x.size()
        avg        = self.avg_pool(x).view([B, 1, C])
        out        = self.conv(avg)
        out        = self.sigmoid(out).view([B, C, 1, 1])
        return x*out

# Common 3×3_Conv Block
def conv3x3(inputchannel, outputchannel, stride=1, groups=1, dilation=1):
    return nn.Conv2d(inputchannel, outputchannel, kernel_size=3, stride=stride,padding=dilation, groups=groups, dilation=dilation, bias=False)

# Common 1×1_Conv Block
def conv1x1(inputchannel, outputchannel, stride=1):
    return nn.Conv2d(inputchannel, outputchannel, kernel_size=1, stride=stride, bias=False)

# Block Type1(Which is commonly used in Less layers ResNet, Such as ResNet18 and ResNet34)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inputchannel, outputchannel, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 3×3 Conv
        self.conv1      = conv3x3(inputchannel, outputchannel, stride)
        self.bn1        = norm_layer(outputchannel)
        self.relu       = nn.ReLU(inplace=True)
        # 3×3 Conv
        self.conv2      = conv3x3(outputchannel, outputchannel)
        self.bn2        = norm_layer(outputchannel)
        # Downsample Section in the end of Block Group.
        self.downsample = downsample
        self.stride     = stride
        self.ECA_block  = ECANet_Block(outputchannel)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ECA_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # Residual Connect
        out += identity
        out = self.relu(out)

        return out

# Block Type2(Which is commonly used in More layers ResNet, Such as ResNet50、ResNet101 and ResNet152)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inputchannel, outputchannel, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.stride     = stride
        if norm_layer is None:
            norm_layer  = nn.BatchNorm2d
        width = int(outputchannel * (base_width / 64.)) * groups
        # 1×1 Conv 
        self.conv1      = conv1x1(inputchannel, width)
        self.bn1        = norm_layer(width)
        # 3×3 Conv
        self.conv2      = conv3x3(width, width, stride, groups, dilation)
        self.bn2        = norm_layer(width)
        # 1x1 Conv
        self.conv3      = conv1x1(width, outputchannel * self.expansion)
        self.bn3        = norm_layer(outputchannel * self.expansion)

        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.ECA_block  = ECANet_Block(outputchannel * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.ECA_block(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # Residual Connect
        out += identity
        out = self.relu(out)

        return out

# Construct the ResNet
class ResNet(nn.Module):
    def __init__(self, block_type, block_config, num_classes=1000, bands=3, downsample_factor=32):
        super(ResNet, self).__init__()
        self.inputchannel    = 64
        self.downsample_factor = downsample_factor
        self.conv1           = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1             = nn.BatchNorm2d(64)
        self.relu            = nn.ReLU(inplace=True)
        self.maxpool         = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
        if downsample_factor == 32:
            self.layer1 = self._make_layer(block_type, 64, block_config[0])
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=2)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=2)
        if downsample_factor == 16:
            self.layer1 = self._make_layer(block_type, 64, block_config[0])
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=2)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=1)
        if downsample_factor == 8:
            self.layer1 = self._make_layer(block_type, 64, block_config[0])
            self.layer2 = self._make_layer(block_type, 128, block_config[1], stride=2)
            self.layer3 = self._make_layer(block_type, 256, block_config[2], stride=1)
            self.layer4 = self._make_layer(block_type, 512, block_config[3], stride=1)  
        

        self.avgpool = nn.AvgPool2d(int(7*(32/downsample_factor)))
        self.fc      = nn.Linear(512 * block_type.expansion, num_classes)

    def _make_layer(self, Block_type, outputchannel, blocks_num, stride=1):
        downsample = None
        if (stride != 1) or (self.inputchannel != outputchannel * Block_type.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.inputchannel, outputchannel * Block_type.expansion, kernel_size=1,  stride=stride, bias=False),
                nn.BatchNorm2d(outputchannel * Block_type.expansion)
        )
        layers = []
        layers.append(Block_type(self.inputchannel, outputchannel, stride, downsample))
        self.inputchannel = outputchannel * Block_type.expansion
        for i in range(1, blocks_num):
            layers.append(Block_type(self.inputchannel, outputchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
         x            = self.conv1(x)
         x            = self.bn1(x)
         x            = self.relu(x)
         x            = self.maxpool(x)
         x            = self.layer1(x)
         x            = self.layer2(x)
         x            = self.layer3(x)
         x            = self.layer4(x)
         x            = self.avgpool(x)
         x            = x.view(x.size(0), -1)
         x            = self.fc(x)
         return x
  

def resnet152(downsample_factor=32, bands=3, num_classes=1000, **kwargs):
    model = ResNet(block_type=Bottleneck, block_config=[3, 8, 36, 3], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)
    return model

def resnet101(downsample_factor=32, bands=3, num_classes=1000, **kwargs):
    model = ResNet(block_type=Bottleneck, block_config=[3, 4, 23, 3], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)
    return model
    
def resnet50(downsample_factor=32, bands=3, num_classes=1000, **kwargs):
    model = ResNet(block_type=Bottleneck, block_config=[3, 4, 6, 3], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)    
    return model

def resnet34(downsample_factor=32, bands=3, num_classes=1000, **kwargs):
    model = ResNet(block_type=BasicBlock, block_config=[3, 4, 6, 3], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)
    return model

def resnet18(downsample_factor=32, bands=3, num_classes=1000, **kwargs):
    model = ResNet(block_type=BasicBlock, block_config=[2, 2, 2, 2], num_classes=num_classes, bands=bands, downsample_factor=downsample_factor)
    return model


if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = resnet50(bands=3, num_classes=1000).to(device)
    x               = torch.randn(2, 3, 224, 224).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (3, 224, 224), batch_dim=0)