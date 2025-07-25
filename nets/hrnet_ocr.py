# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/7/1} ${21:51}
# @Function   : function to 
# @Description: 


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale   = scale

    def forward(self, feats, probs):
        b, c, h, w  = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs       = probs.view(b, c, -1)
        feats       = feats.view(b, feats.size(1), -1)
        feats       = feats.permute(0, 2, 1)  # b x hw x c 
        probs       = F.softmax(self.scale * probs, dim=2)  # b x k x hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)  # b x k x c
        return ocr_context

class _ObjectAttentionBlock(nn.Module):
    def __init__(self, in_channels, key_channels, scale=1):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale        = scale
        self.in_channels  = in_channels
        self.key_channels = key_channels
        self.pool         = nn.MaxPool2d(kernel_size=(scale, scale))

        self.f_pixel      = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        # Object feature transform
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(key_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels, key_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        
        self.f_up = nn.Sequential(
            nn.Conv2d(key_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, proxy):
        b, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # Query: pixel features
        query = self.f_pixel(x).view(b, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        
        # Key: object features
        key   = self.f_object(proxy).view(b, self.key_channels, -1)
        
        # Value: downsampled object features
        value = self.f_down(proxy).view(b, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        # Similarity map
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # Context aggregation
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(b, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        
        if self.scale > 1:
            context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock2D, self).__init__(in_channels, key_channels, scale)

class SpatialOCR_Module(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, key_channels, scale)
        self.conv_bn_dropout      = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output  = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output

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
        self.num_inchannels     = num_inchannels
        self.fuse_method        = fuse_method
        self.num_branches       = num_branches
        self.multi_scale_output = multi_scale_output
        self.branches           = self._make_branches(num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers        = self._make_fuse_layers()
        self.relu               = nn.ReLU(inplace=True)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels, stride=1):
        downsample = None
        if stride != 1 or self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index], num_channels[branch_index] * block.expansion,
				          kernel_size=1, stride=stride, bias=False),
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

        num_branches   = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers    = []
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
                                nn.Conv2d(num_inchannels[j], num_outchannels_conv3x3, 3, 2, 1, bias=False),
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

class HighResolutionNetOCR(nn.Module):
    def __init__(self, bands=3, num_classes=1000, backbone=32, ocr_mid_channels=512, ocr_key_channels=256, full_resolution=False):
        super(HighResolutionNetOCR, self).__init__()
        self.full_resolution = full_resolution
        
        # 骨干网络配置
        cfg = {
            18: {
                'STAGE1': {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4]},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'}
            },
            32: {
                'STAGE1': {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4]},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [32, 64], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [32, 64, 128], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [32, 64, 128, 256], 'FUSE_METHOD': 'SUM'}
            },
            48: {
                'STAGE1': {'NUM_CHANNELS': [64], 'BLOCK': 'BOTTLENECK', 'NUM_BLOCKS': [4]},
                'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4], 'NUM_CHANNELS': [48, 96], 'FUSE_METHOD': 'SUM'},
                'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4], 'NUM_CHANNELS': [48, 96, 192], 'FUSE_METHOD': 'SUM'},
                'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 
                           'NUM_BLOCKS': [4, 4, 4, 4], 'NUM_CHANNELS': [48, 96, 192, 384], 'FUSE_METHOD': 'SUM'}
            }
        }
        self.model_cfgs = cfg[backbone]
        
        # 初始化stem网络
        self.conv1 = nn.Conv2d(bands, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # Stage 1
        stage1_cfg         = self.model_cfgs['STAGE1']
        num_channels       = stage1_cfg['NUM_CHANNELS'][0]
        block              = blocks_dict[stage1_cfg['BLOCK']]
        num_blocks         = stage1_cfg['NUM_BLOCKS'][0]
        self.layer1        = self._make_layer(block, 64, num_channels, num_blocks)
        stage1_out_channel = block.expansion * num_channels

        # Stage 2
        stage2_cfg       = self.model_cfgs['STAGE2']
        num_channels     = stage2_cfg['NUM_CHANNELS']
        block            = blocks_dict[stage2_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(stage2_cfg, num_channels)

        # Stage 3
        stage3_cfg       = self.model_cfgs['STAGE3']
        num_channels     = stage3_cfg['NUM_CHANNELS']
        block            = blocks_dict[stage3_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(stage3_cfg, num_channels)

        # Stage 4
        stage4_cfg       = self.model_cfgs['STAGE4']
        num_channels     = stage4_cfg['NUM_CHANNELS']
        block            = blocks_dict[stage4_cfg['BLOCK']]
        num_channels     = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(stage4_cfg, num_channels, multi_scale_output=True)

        # OCR模块
        last_inp_channels = int(np.sum(pre_stage_channels))
        
        # 3x3卷积用于特征变换
        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(last_inp_channels, ocr_mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        
        # 空间聚合模块
        self.ocr_gather_head = SpatialGather_Module(cls_num=num_classes)
        
        # OCR分布头
        self.ocr_distri_head = SpatialOCR_Module(
            in_channels  = ocr_mid_channels,
            key_channels = ocr_key_channels,
            out_channels = ocr_mid_channels,
            scale        = 1,
            dropout      = 0.05
        )
        
        # 主分类头
        self.cls_head = nn.Conv2d(ocr_mid_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        
        # 辅助分类头
        self.aux_head = nn.Sequential(
            nn.Conv2d(last_inp_channels, last_inp_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_inp_channels, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
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

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules  = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks   = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block        = blocks_dict[layer_config['BLOCK']]
        fuse_method  = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
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
        
        # Stem网络
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # Stage 2
        x_list = []
        for i in range(self.model_cfgs['STAGE2']['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        # Stage 3
        x_list = []
        for i in range(self.model_cfgs['STAGE3']['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                if i < self.model_cfgs['STAGE2']['NUM_BRANCHES']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # Stage 4
        x_list = []
        for i in range(self.model_cfgs['STAGE4']['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                if i < self.model_cfgs['STAGE3']['NUM_BRANCHES']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        stage4_output = self.stage4(x_list)

        # 上采样所有分支到最高分辨率
        x0_h, x0_w = stage4_output[0].size(2), stage4_output[0].size(3)
        x1 = F.interpolate(stage4_output[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(stage4_output[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(stage4_output[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        feats = torch.cat([stage4_output[0], x1, x2, x3], 1)

        # OCR模块处理
        out_aux = self.aux_head(feats)
        feats   = self.conv3x3_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats   = self.ocr_distri_head(feats, context)
        out     = self.cls_head(feats)
        
        # 如果需要原始分辨率输出
        if self.full_resolution:
            out_aux = F.interpolate(out_aux, size=input_size, mode='bilinear', align_corners=True)
            out     = F.interpolate(out, size=input_size, mode='bilinear', align_corners=True)
        
        #return [out_aux, out]
        return out


def hrnetocr(bands=3, num_classes=1000, backbone=48, ocr_mid_channels=512, ocr_key_channels=256, full_resolution=True):
    model = HighResolutionNetOCR(
        bands            = bands,
        num_classes      = num_classes,
        backbone         = backbone,
        ocr_mid_channels = ocr_mid_channels,
        ocr_key_channels = ocr_key_channels,
        full_resolution  = full_resolution)
    return model

if __name__ == '__main__':
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = hrnetocr(bands=3,  num_classes=19, backbone=48).to(device)
    x               = torch.randn(2, 3, 256, 256).to(device)
    aux_out, out    =  model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs(G): ', (flops/1e9/x.shape[0]), 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(out.shape))
    summary(model, (3, 256, 256), batch_dim=0)