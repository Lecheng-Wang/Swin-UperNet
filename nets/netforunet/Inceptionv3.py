# encoding = utf-8

# @Author  ：Lecheng Wang
# @Time    : ${DATE} ${TIME}
# @Function: Realization of inceptionv3 architecture

import torch
import torch.nn as nn
from torchsummary import summary
from thop import profile

class BasicConv2d(nn.Module):
    def __init__(self, inputchannel, outputchannel, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(inputchannel, outputchannel, bias=False, **kwargs)
        self.bn   = nn.BatchNorm2d(outputchannel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class InceptionA(nn.Module):
    def __init__(self, inputchannel, pool_features):
        super(InceptionA, self).__init__()
        self.branch1x1 = BasicConv2d(inputchannel, 64, kernel_size=1)
        self.branch5x5 = nn.Sequential(
            BasicConv2d(inputchannel, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5, padding=2)
        )
        self.branch3x3 = nn.Sequential(
            BasicConv2d(inputchannel, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, padding=1)
        )
        self.branchpool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inputchannel, pool_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        branch1x1  = self.branch1x1(x)
        branch5x5  = self.branch5x5(x)
        branch3x3  = self.branch3x3(x)
        branchpool = self.branchpool(x)
        outputs    = [branch1x1, branch5x5, branch3x3, branchpool]
        return torch.cat(outputs, 1)

class InceptionB(nn.Module):
    def __init__(self, inputchannel):
        super(InceptionB, self).__init__()
        self.branch3x3      = BasicConv2d(inputchannel, 384, kernel_size=3, stride=2, padding=1)
        self.branch3x3stack = nn.Sequential(
            BasicConv2d(inputchannel, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3, padding=1),
            BasicConv2d(96, 96, kernel_size=3, stride=2, padding=1)
        )
        self.branchpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3      = self.branch3x3(x)
        branch3x3stack = self.branch3x3stack(x)
        branchpool     = self.branchpool(x)
        outputs        = [branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)



class InceptionC(nn.Module):
    def __init__(self, inputchannel, channels_7x7):
        super(InceptionC, self).__init__()
        self.branch1x1 = BasicConv2d(inputchannel, 192, kernel_size=1)
        c7 = channels_7x7
        self.branch7x7 = nn.Sequential(
            BasicConv2d(inputchannel, c7, kernel_size=1),
            BasicConv2d(c7, c7,  kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch7x7stack = nn.Sequential(
            BasicConv2d(inputchannel, c7, kernel_size=1),
            BasicConv2d(c7, c7,  kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, c7,  kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(c7, c7,  kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(c7, 192, kernel_size=(1, 7), padding=(0, 3))
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inputchannel, 192, kernel_size=1),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7stack = self.branch7x7stack(x)
        branchpool = self.branch_pool(x)
        outputs = [branch1x1, branch7x7, branch7x7stack, branchpool]
        return torch.cat(outputs, 1)

class InceptionD(nn.Module):
    def __init__(self, inputchannel):
        super(InceptionD, self).__init__()
        self.branch3x3 = nn.Sequential(
            BasicConv2d(inputchannel, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2, padding=1)
        )
        self.branch7x7 = nn.Sequential(
            BasicConv2d(inputchannel, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(192, 192, kernel_size=3, stride=2, padding=1)
        )
        self.branchpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7 = self.branch7x7(x)
        branchpool = self.branchpool(x)
        outputs = [branch3x3, branch7x7, branchpool]
        return torch.cat(outputs, 1)

class InceptionE(nn.Module):
    def __init__(self, inputchannel):
        super(InceptionE, self).__init__()
        self.branch1x1         = BasicConv2d(inputchannel, 320, kernel_size=1)
        self.branch3x3_1       = BasicConv2d(inputchannel, 384, kernel_size=1)
        self.branch3x3_2a      = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b      = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch3x3stack_1  = BasicConv2d(inputchannel, 448, kernel_size=1)
        self.branch3x3stack_2  = BasicConv2d(448, 384, kernel_size=3, padding=1)
        self.branch3x3stack_3a = BasicConv2d(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3stack_3b = BasicConv2d(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inputchannel, 192, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3)
        ]
        branch3x3      = torch.cat(branch3x3, 1)
        branch3x3stack = self.branch3x3stack_1(x)
        branch3x3stack = self.branch3x3stack_2(branch3x3stack)
        branch3x3stack = [
            self.branch3x3stack_3a(branch3x3stack),
            self.branch3x3stack_3b(branch3x3stack)
        ]
        branch3x3stack = torch.cat(branch3x3stack, 1)
        branchpool     = self.branch_pool(x)
        outputs        = [branch1x1, branch3x3, branch3x3stack, branchpool]
        return torch.cat(outputs, 1)

class InceptionV3(nn.Module):
    def __init__(self, bands=3, num_classes=1000):
        super(InceptionV3, self).__init__()
        #self.Conv2d_1a_3x3 = BasicConv2d(bands, 32, kernel_size=3, stride=2, padding=1)
        self.Conv2d_1a_3x3 = BasicConv2d(bands, 32, kernel_size=3, stride=1, padding=1) # used in deeplabv3+
        self.Conv2d_2a_3x3 = BasicConv2d(32, 32,  kernel_size=3, stride=1, padding=1)
        self.Conv2d_2b_3x3 = BasicConv2d(32, 64,  kernel_size=3, stride=2, padding=1)
        self.Conv2d_3b_1x1 = BasicConv2d(64, 80,  kernel_size=3, padding=1)
        self.Conv2d_4a_3x3 = BasicConv2d(80, 192, kernel_size=3, stride=2, padding=1)

        self.Mixed_5b = InceptionA(192, pool_features=32)
        self.Mixed_5c = InceptionA(256, pool_features=64)
        self.Mixed_5d = InceptionA(288, pool_features=64)

        self.Mixed_6a = InceptionB(288)
        self.Mixed_6b = InceptionC(768, channels_7x7=128)
        self.Mixed_6c = InceptionC(768, channels_7x7=160)
        self.Mixed_6d = InceptionC(768, channels_7x7=160)
        self.Mixed_6e = InceptionC(768, channels_7x7=192)

        self.Mixed_7a = InceptionD(768)
        self.Mixed_7b = InceptionE(1280)
        self.Mixed_7c = InceptionE(2048)

#        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#        self.dropout = nn.Dropout2d()
#        self.linear  = nn.Linear(2048, num_classes)

        self._initialize_weights()
    def forward(self, x):
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        low_level_feature = x   # used in deeplabv3+
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)

        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)

        high_level_feature = x  # used in deeplabV3+

#        x = self.avgpool(x)
#        x = self.dropout(x)
#        x = x.view(x.size(0), -1)
#        x = self.linear(x)
#        return x
        return low_level_feature, high_level_feature   # used in deeplabv3+

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# Test Model Structure and Outputsize
if __name__ == "__main__":
    model         = InceptionV3(bands=3)
    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    x             = torch.randn(1, 3, 224, 224).to(device)  # Assume inputsize 3×224×224 RGB image
    print("Input shape:", x.shape)
    output        = model(x)
    flops, params = profile(model, inputs=(x, ))
    print('flops: ', flops, 'params: ', params)
    print("Output shape:", output[0].shape, output[1].shape)
    summary(model, (3, 224, 224))