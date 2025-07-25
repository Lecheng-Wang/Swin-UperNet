


import torch
import torch.nn            as nn
import torch.nn.functional as F
import numpy               as np

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1      = conv3x3(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm2d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.conv2      = conv3x3(planes, planes)
        self.bn2        = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes * self.expansion)
        self.relu       = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self.num_inchannels = num_inchannels
        self.fuse_method    = fuse_method
        self.num_branches   = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches    = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu        = nn.ReLU(inplace=True)


    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion),
            )
        layers = []
        layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index], stride, downsample))
        self.num_inchannels[branch_index] = num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index], num_channels[branch_index]))
        return nn.Sequential(*layers)


    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))
        return nn.ModuleList(branches)


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j], num_inchannels[i], 1, 1, 0, bias=False),
                        nn.BatchNorm2d(num_inchannels[i])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3),
                                nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(self.fuse_layers[i][j](x[j]), size=[height_output, width_output], mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HighResolutionNet(nn.Module):

    def __init__(self, bands=3, num_classes=1000, backbone=32, version='v2', full_resolution=False, **kwargs):
        super(HighResolutionNet, self).__init__()
        self.version         = version  # 'v1' or 'v2'
        self.full_resolution = full_resolution
        cfg = {
            18: {'num_module':[1,1,4,3], 'num_branch':[1,2,3,4], 'num_channels':[18,36,72, 144], 'num_block': [4,4,4,4]},
            32: {'num_module':[1,1,4,3], 'num_branch':[1,2,3,4], 'num_channels':[32,64,128,256], 'num_block': [4,4,4,4]},
            48: {'num_module':[1,1,4,3], 'num_branch':[1,2,3,4], 'num_channels':[48,96,192,384], 'num_block': [4,4,4,4]}
        }
        self.model_cfgs = cfg[backbone]
        
        # stem net
        self.conv1 = nn.Conv2d(bands, 64, kernel_size=3, stride=2, padding=1,bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)


        num_channels       = 64
        block              = blocks_dict['BOTTLENECK']
        num_blocks         = 4
        self.layer1        = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion*num_channels

        num_channels       = self.model_cfgs['num_channels'][:2]
        block              = blocks_dict['BASIC']
        num_channels       = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1   = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
                       self.model_cfgs['num_module'][1], self.model_cfgs['num_branch'][1],
                       self.model_cfgs['num_block'][:2], self.model_cfgs['num_channels'][:2], block, num_channels)


        num_channels       = self.model_cfgs['num_channels'][:3]
        block              = blocks_dict['BASIC']
        num_channels       = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2   = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
                       self.model_cfgs['num_module'][2], self.model_cfgs['num_branch'][2],
                       self.model_cfgs['num_block'][:3], self.model_cfgs['num_channels'][:3], block, num_channels)


        num_channels       = self.model_cfgs['num_channels'][:4]
        block              = blocks_dict['BASIC']
        num_channels       = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3   = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
                       self.model_cfgs['num_module'][3], self.model_cfgs['num_branch'][3],
                       self.model_cfgs['num_block'][:4], self.model_cfgs['num_channels'][:4], 
                       block, num_channels, multi_scale_output=True if self.version == 'v2' else False)
        
        if self.version == 'v2':
            last_inp_channels = int(np.sum(pre_stage_channels))
            self.last_layer = nn.Sequential(
                nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(last_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.last_layer = nn.Sequential(
                nn.Conv2d(in_channels=pre_stage_channels[0], out_channels=pre_stage_channels[0], kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(pre_stage_channels[0]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=pre_stage_channels[0], out_channels=num_classes, kernel_size=1, stride=1, padding=0)
            )

        if self.full_resolution:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                # 可选：添加额外的卷积层来细化上采样结果
                # nn.Conv2d(NUM_CLASSES, NUM_CLASSES, kernel_size=3, padding=1),
                # nn.ReLU(inplace=True)
            )

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i], num_channels_cur_layer[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i]),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] if j == i-num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(inchannels, outchannels, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(outchannels),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, num_modules,num_branches,num_blocks,num_channels,block,num_inchannels,fuse_method='SUM',multi_scale_output=True):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches, block, num_blocks, num_inchannels, num_channels, fuse_method, reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.model_cfgs['num_branch'][1]):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.model_cfgs['num_branch'][2]):
            if self.transition2[i] is not None:
                if i < self.model_cfgs['num_branch'][1]:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        y_list = self.stage3(x_list)
        x_list = []

        for i in range(self.model_cfgs['num_branch'][3]):
            if self.transition3[i] is not None:
                if i < self.model_cfgs['num_branch'][2]:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        stage4_output = self.stage4(x_list)

        if self.version == 'v2':
            x0_h, x0_w = stage4_output[0].size(2), stage4_output[0].size(3)
            x1 = F.interpolate(stage4_output[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x2 = F.interpolate(stage4_output[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x3 = F.interpolate(stage4_output[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
            x  = torch.cat([stage4_output[0], x1, x2, x3], 1)
            x  = self.last_layer(x)
        else:
            x  = self.last_layer(stage4_output[0])

        if self.full_resolution:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x

def hrnet(bands=3, num_classes=1000, backbone=48, version='v2',**kwargs):
    model = HighResolutionNet(bands=bands, num_classes=num_classes, backbone=backbone, version=version, full_resolution=True, **kwargs)
    return model

if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = hrnet(bands=6,  num_classes=3, backbone=18, version='v2').to(device)
    x               = torch.randn(2, 6, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs(G): ', (flops/1e9/x.shape[0]), 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, (6, 256, 256), batch_dim=0)