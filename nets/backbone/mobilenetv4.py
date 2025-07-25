# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : ${2025/7/2} ${19:00}
# @Function   : mobilenetv4
# @Description: 

import torch
import torch.nn as nn
import torch.nn.functional as F

bands=3

MNV4ConvSmall_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [bands, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 32, 3, 2],
            [32, 32, 1, 1]
        ]
    },
    "layer2": {
        "block_name": "convbn",
        "num_blocks": 2,
        "block_specs": [
            [32, 96, 3, 2],
            [96, 64, 1, 1]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [64, 96, 5, 5, True, 2, 3, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 0, 3, True, 1, 2, False],
            [96, 96, 3, 0, True, 1, 4, False],
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 6,
        "block_specs": [
            [96,  128, 3, 3, True, 2, 6, False],
            [128, 128, 5, 5, True, 1, 4, False],
            [128, 128, 0, 5, True, 1, 4, False],
            [128, 128, 0, 5, True, 1, 3, False],
            [128, 128, 0, 3, True, 1, 4, False],
            [128, 128, 0, 3, True, 1, 4, False],
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [128, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4ConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [bands, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4, False],
            [80, 80, 3, 3, True, 1, 2, False]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80,  160, 3, 5, True, 2, 6, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 5, True, 1, 4, False],
            [160, 160, 3, 3, True, 1, 4, False],
            [160, 160, 3, 0, True, 1, 4, False],
            [160, 160, 0, 0, True, 1, 2, False],
            [160, 160, 3, 0, True, 1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6, False],
            [256, 256, 5, 5, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 3, 0, True, 1, 4, False],
            [256, 256, 3, 5, True, 1, 2, False],
            [256, 256, 5, 5, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 0, 0, True, 1, 4, False],
            [256, 256, 5, 0, True, 1, 2, False]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [256, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4ConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [bands, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4],
            [96, 96, 3, 3, True, 1, 4]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96,  192, 3, 5, True, 2, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 3, True, 1, 4, False],
            [192, 192, 3, 5, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 5, 3, True, 1, 4, False],
            [192, 192, 3, 0, True, 1, 4, False]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 13,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 3, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 3, True, 1, 4, False],
            [512, 512, 5, 5, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False],
            [512, 512, 5, 0, True, 1, 4, False]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [512, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


def mhsa(num_heads, key_dim, value_dim, px):
    if px == 24:
        kv_strides = 2
    elif px == 12:
        kv_strides = 1
    query_h_strides = 1
    query_w_strides = 1
    use_layer_scale = True
    use_multi_query = True
    use_residual = True
    return [
        num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides,
        use_layer_scale, use_multi_query, use_residual
    ]


MNV4HybridConvMedium_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [bands, 32, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [32, 48, 2, 4.0, False]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 80, 3, 5, True, 2, 4, True],
            [80, 80, 3, 3, True, 1, 2, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 8,
        "block_specs": [
            [80,  160, 3, 5, True, 2, 6, True],
            [160, 160, 0, 0, True, 1, 2, True],
            [160, 160, 3, 3, True, 1, 4, True],
            [160, 160, 3, 5, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 3, True, 1, 4, True, mhsa(4, 64, 64, 24)],
            [160, 160, 3, 0, True, 1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 12,
        "block_specs": [
            [160, 256, 5, 5, True, 2, 6, True],
            [256, 256, 5, 5, True, 1, 4, True],
            [256, 256, 3, 5, True, 1, 4, True],
            [256, 256, 3, 5, True, 1, 4, True],
            [256, 256, 0, 0, True, 1, 2, True],
            [256, 256, 3, 5, True, 1, 2, True],
            [256, 256, 0, 0, True, 1, 2, True],
            [256, 256, 0, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 3, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 5, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, True, mhsa(4, 64, 64, 12)],
            [256, 256, 5, 0, True, 1, 4, True]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [256, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}


MNV4HybridConvLarge_BLOCK_SPECS = {
    "conv0": {
        "block_name": "convbn",
        "num_blocks": 1,
        "block_specs": [
            [bands, 24, 3, 2]
        ]
    },
    "layer1": {
        "block_name": "fused_ib",
        "num_blocks": 1,
        "block_specs": [
            [24, 48, 2, 4.0, False, True]
        ]
    },
    "layer2": {
        "block_name": "uib",
        "num_blocks": 2,
        "block_specs": [
            [48, 96, 3, 5, True, 2, 4, True],
            [96, 96, 3, 3, True, 1, 4, True]
        ]
    },
    "layer3": {
        "block_name": "uib",
        "num_blocks": 11,
        "block_specs": [
            [96,  192, 3, 5, True, 2, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 3, True, 1, 4, True],
            [192, 192, 3, 5, True, 1, 4, True],
            [192, 192, 5, 3, True, 1, 4, True],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 5, 3, True, 1, 4, True, mhsa(8, 48, 48, 24)],
            [192, 192, 3, 0, True, 1, 4, True]
        ]
    },
    "layer4": {
        "block_name": "uib",
        "num_blocks": 14,
        "block_specs": [
            [192, 512, 5, 5, True, 2, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 3, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 0, True, 1, 4, True],
            [512, 512, 5, 3, True, 1, 4, True],
            [512, 512, 5, 5, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True, mhsa(8, 64, 64, 12)],
            [512, 512, 5, 0, True, 1, 4, True]
        ]
    },  
    "layer5": {
        "block_name": "convbn",
        "num_blocks": 3,
        "block_specs": [
            [512, 960, 1, 1],
            "AdaptiveAvgPool2d",
            [960, 1280, 1, 1]
        ]
    }
}

MODEL_SPECS = {
    "MobileNetV4ConvSmall": MNV4ConvSmall_BLOCK_SPECS,
    "MobileNetV4ConvMedium": MNV4ConvMedium_BLOCK_SPECS,
    "MobileNetV4ConvLarge": MNV4ConvLarge_BLOCK_SPECS,
    "MobileNetV4HybridMedium": MNV4HybridConvMedium_BLOCK_SPECS,
    "MobileNetV4HybridLarge": MNV4HybridConvLarge_BLOCK_SPECS
}

def make_divisible(value: float, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)

    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return int(new_value)

def conv_2d(inp, oup, kernel_size=3, stride=1, groups=1, bias=False, norm=True, act=True):
    conv    = nn.Sequential()
    padding = (kernel_size - 1) // 2
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.ReLU())
    return conv

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, act=False, squeeze_excitation=False):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(round(inp * expand_ratio))
        self.block = nn.Sequential()
        if expand_ratio != 1:
            self.block.add_module('exp_1x1', conv_2d(inp, hidden_dim, kernel_size=3, stride=stride))
        if squeeze_excitation:
            self.block.add_module('conv_3x3', conv_2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, groups=hidden_dim))
        self.block.add_module('red_1x1', conv_2d(hidden_dim, oup, kernel_size=1, stride=1, act=act))
        self.use_res_connect = self.stride == 1 and inp == oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class UniversalInvertedBottleneckBlock(nn.Module):
    def __init__(self, inp, oup, start_dw_kernel_size, middle_dw_kernel_size, middle_dw_downsample, 
                       stride, expand_ratio, use_layer_scale=False):
        super().__init__()
        self.use_res_connect = (inp == oup and stride == 1)

        # Starting depthwise conv.
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:            
            stride_ = stride if not middle_dw_downsample else 1
            self._start_dw_ = conv_2d(inp, inp, kernel_size=start_dw_kernel_size, stride=stride_, groups=inp, act=False)
        # Expansion with 1x1 convs.
        expand_filters = make_divisible(inp * expand_ratio, 8)
        self._expand_conv = conv_2d(inp, expand_filters, kernel_size=1)
        # Middle depthwise conv.
        self.middle_dw_kernel_size = middle_dw_kernel_size
        if self.middle_dw_kernel_size:
            stride_ = stride if middle_dw_downsample else 1
            self._middle_dw = conv_2d(expand_filters, expand_filters, kernel_size=middle_dw_kernel_size, stride=stride_, groups=expand_filters)
        # Projection with 1x1 convs.
        self._proj_conv = conv_2d(expand_filters, oup, kernel_size=1, stride=1, act=False)
        
        # Ending depthwise conv.
        self._use_layer_scale = use_layer_scale
        if self._use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale            = MNV4LayerScale(oup, self.layer_scale_init_value)

    def forward(self, x):
        shortcut = x
        if self.start_dw_kernel_size:
            x = self._start_dw_(x)
        x = self._expand_conv(x)

        if self.middle_dw_kernel_size:
            x = self._middle_dw(x)
        x = self._proj_conv(x)

        if self._use_layer_scale:
            x = self.layer_scale(x)

        if self.use_res_connect:
            x = x + shortcut

        return x

class MultiQueryAttentionLayerWithDownSampling(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides, dw_kernel_size=3, dropout=0.0):
        super().__init__()
        self.num_heads       = num_heads
        self.key_dim         = key_dim
        self.value_dim       = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides      = kv_strides
        self.dw_kernel_size  = dw_kernel_size
        self.dropout         = dropout
        self.head_dim        = key_dim // num_heads

        if self.query_h_strides > 1 or self.query_w_strides > 1:
            self._query_downsampling_norm = nn.BatchNorm2d(inp)
        self._query_proj     = conv_2d(inp, num_heads*key_dim, 1, 1, norm=False, act=False)
       
        self.key   = nn.Sequential()
        self.value = nn.Sequential()
        if self.kv_strides > 1:
            self.key.add_module('_key_dw_conv', conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False))
            self.value.add_module('_value_dw_conv', conv_2d(inp, inp, dw_kernel_size, kv_strides, groups=inp, norm=True, act=False))
        self.key.add_module('_key_proj', conv_2d(inp, key_dim, 1, 1, norm=False, act=False))
        self.value.add_module('_value_proj', conv_2d(inp, key_dim, 1, 1, norm=False, act=False))

        self._output_proj = conv_2d(num_heads*key_dim, inp, 1, 1, norm=False, act=False)
        self.dropout = nn.Dropout(p=dropout)

    def _reshape_projected_query(self, t: torch.Tensor, num_heads: int, key_dim: int):
        """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
        s = t.shape
        t = t.reshape(s[0], num_heads, key_dim, -1)
        return t.transpose(-1, -2).contiguous()
       
    def _reshape_input(self, t: torch.Tensor):
        """Reshapes a tensor to three dimensions, keeping the batch and channels."""
        s = t.shape
        t = t.reshape(s[0], s[1], -1).transpose(1, 2)
        return t.unsqueeze(1).contiguous()

    def _reshape_output(self, t: torch.Tensor, num_heads: int, h_px: int, w_px: int):
        """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
        s = t.shape
        feat_dim = s[-1] * num_heads
        t = t.transpose(1, 2)
        return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()
       
    def forward(self, x):
        batch_size, seq_length, H, W = x.size()
        if self.query_h_strides > 1 or self.query_w_strides > 1:
            q = F.avg_pool2d(self.query_h_stride, self.query_w_stride)
            q = self._query_downsampling_norm(q)
            q = self._query_proj(q)
        else:
            q = self._query_proj(x)
        # desired q shape: [b, h, k, n x n] - [b, l, h, k]
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)
        k = self.key(x)
        # output shape of k: [b, k, p], p = m x m
        k = self._reshape_input(k)
        v = self.value(x)  
        # output shape of v: [ b, p, k], p = m x m
        v = self._reshape_input(v)

        # calculate attn score
        q          = q * (self.key_dim ** -0.5)
        attn_score = q @ k.transpose(-1, -2)
        attn_score = attn_score.softmax(dim=-1)
        attn_score = self.dropout(attn_score)
        output     = attn_score @ v

        # reshape o into [b, hk, n, n,]
        output = self._reshape_output(output, self.num_heads, H // self.query_h_strides, W // self.query_w_strides)
        output = self._output_proj(output)
        return output
   
class MNV4LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x * gamma


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides,
                       kv_strides, use_layer_scale, use_multi_query, use_residual = True):
        super().__init__()
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides      = kv_strides
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual    = use_residual

        self._input_norm = nn.BatchNorm2d(inp)
        if self.use_multi_query:
            self.multi_query_attention = MultiQueryAttentionLayerWithDownSampling(
                inp, num_heads, key_dim, value_dim, query_h_strides, query_w_strides, kv_strides
            )
        else:
            self.multi_head_attention = nn.MultiheadAttention(inp, num_heads, kdim=key_dim)
       
        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale            = MNV4LayerScale(inp, self.layer_scale_init_value)
   
    def forward(self, x):
        # Not using CPE, skipped
        # input norm
        shortcut = x
        x = self._input_norm(x)
        # multi query
        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x)
        # layer scale
        if self.use_layer_scale:
            x = self.layer_scale(x)
        # use residual
        if self.use_residual:
            x = x + shortcut
        return x

def build_blocks(layer_spec):
    if not layer_spec.get('block_name'):
        return nn.Sequential()
    block_names = layer_spec['block_name']
    layers      = nn.Sequential()
    if block_names == "convbn":
        schema_ = ['inp', 'oup', 'kernel_size', 'stride']
        for i in range(layer_spec['num_blocks']):
            specs_ = layer_spec['block_specs'][i]
            if isinstance(specs_, list):
                args = dict(zip(schema_, layer_spec['block_specs'][i]))
                layers.add_module(f"convbn_{i}", conv_2d(**args))
            elif specs_ == "AdaptiveAvgPool2d":
                layers.add_module(f"AdaptiveAvgPool2d", nn.AdaptiveAvgPool2d(1))
            else:
                raise NotImplementedError
    elif block_names == "uib":
        schema_ =  ['inp', 'oup', 'start_dw_kernel_size', 'middle_dw_kernel_size', 'middle_dw_downsample', 'stride', 'expand_ratio', 'use_layer_scale', 'mhsa']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            mhsa = args.pop("mhsa") if "mhsa" in args else 0
            layers.add_module(f"uib_{i}", UniversalInvertedBottleneckBlock(**args))
            if mhsa:
                mhsa_schema_ = [
                    "inp", "num_heads", "key_dim", "value_dim", "query_h_strides", "query_w_strides", "kv_strides", 
                    "use_layer_scale", "use_multi_query", "use_residual"
                ]
                args = dict(zip(mhsa_schema_, [args['oup']] + (mhsa)))
                layers.add_module(f"mhsa_{i}", MultiHeadSelfAttentionBlock(**args))
    elif block_names == "fused_ib":
        schema_ = ['inp', 'oup', 'stride', 'expand_ratio', 'act']
        for i in range(layer_spec['num_blocks']):
            args = dict(zip(schema_, layer_spec['block_specs'][i]))
            layers.add_module(f"fused_ib_{i}", InvertedResidual(**args))
    else:
        raise NotImplementedError
    return layers


class MobileNetV4(nn.Module):
    def __init__(self, num_classes=1000, backbone='MobileNetV4ConvSmall'): # MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge
        super().__init__()
        assert backbone in MODEL_SPECS.keys()
        self.model   = backbone
        self.spec    = MODEL_SPECS[self.model]
       
        # conv0
        self.conv0   = build_blocks(self.spec['conv0'])
        # layer1
        self.layer1  = build_blocks(self.spec['layer1'])
        # layer2
        self.layer2  = build_blocks(self.spec['layer2'])
        # layer3
        self.layer3  = build_blocks(self.spec['layer3'])
        # layer4
        self.layer4  = build_blocks(self.spec['layer4'])
        # layer5   
        self.layer5  = build_blocks(self.spec['layer5'])

        self.flatten = nn.Flatten()
        self.linear  = nn.Linear(1280, num_classes, bias=True)
               
    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)
        
        out = self.flatten(x5)
        out = self.linear(out)
        return out  

def mobilenetv4(num_classes=1000):
    # MobileNetV4ConvSmall, MobileNetV4ConvMedium, MobileNetV4ConvLarge
    # MobileNetV4HybridMedium, MobileNetV4HybridLarge
    model = MobileNetV4(num_classes=num_classes, backbone='MobileNetV4ConvMedium') 
    return model

if __name__ == "__main__":
    from torchinfo  import summary
    from thop       import profile
    device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model           = mobilenetv4(num_classes=1000).to(device)
    x               = torch.randn(2, 3, 256, 256).to(device)
    output          = model(x)
    flops, params   = profile(model, inputs=(x, ), verbose=False)

    print('GFLOPs: ', (flops/1e9)/x.shape[0], 'Params(M): ', params/1e6)
    print("Input  shape:", list(x.shape))
    print("Output shape:", list(output.shape))
    summary(model, input_size=(2, 3, 256, 256), device=device.type)


