import math
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import tqdm
from torch import optim
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from basicsr.utils.registry import ARCH_REGISTRY



NEG_INF = -1000000



############################


class CFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 5, 1,
                                2, bias=True, groups=hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


######################################################
class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr=False, compress_ratio=3, squeeze_factor=15):
        super(CAB, self).__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // 3, 3, 1, 1, groups=num_feat // 6),
            nn.GELU(),
            nn.Conv2d(num_feat // 3, num_feat, 3, 1, 1, groups=num_feat // 6),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# global context estimation module
class GCE(nn.Module):
    def __init__(self, dim, down_scale=8):
        super(GCE, self).__init__()
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.conv_1 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.gelu = nn.GELU()
        self.down_scale = down_scale
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        _, _, h, w = x.shape
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        # scale and shift
        enhanced_features = x_s * self.alpha + x_v * self.belt
        # nearst upsampling is harmful for the performance
        x_l = x * F.interpolate(self.gelu(self.conv_1(enhanced_features)),
                                size=(h, w), align_corners=False, mode='bilinear')
        return x_l

# cross-scale overlapping fusion module
class COFB(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU())
        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 15, 1, 15 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        )
        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x


# local aggregation module
class LA(nn.Sequential):

    def __init__(self, dim: int, mlp_ration=2):
        super().__init__(
            nn.Conv2d(dim, int(dim * mlp_ration), 1),  # Expand channels
            nn.GELU(),
            # Depthwise convolution with group
            nn.Conv2d(int(dim * mlp_ration), int(dim * mlp_ration), 3, 1, 1,
                      groups=int(dim * mlp_ration) // 6),
            nn.GELU(),
            nn.Conv2d(int(dim * mlp_ration), dim, 1)  # Reduce channels
        )
        trunc_normal_(self[-1].weight, std=0.02)


########################################
# ISB renamed to MRFE
class InceptionStyleDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, branch_ratio=4, kernel_sizes=[0, 5, 11, 17]):
        super().__init__()
        assert in_channels % branch_ratio == 0, "in_channels must be divisible by branch_ratio"
        gc = int(in_channels // 4)  # group channel numbers of a convolution branch
        kernel1 = kernel_sizes[1]
        kernel2 = kernel_sizes[2]
        kernel3 = kernel_sizes[3]

        self.dwconv_hw = nn.Conv2d(gc, gc, kernel_size=kernel1, padding=kernel1 // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=kernel2, padding=kernel2 // 2, groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=kernel3, padding=kernel3 // 2, groups=gc)

        self.split_indexes = (gc, gc, gc, gc)

    # input B C H W
    def forward(self, x):
        x_id, x_5, x_11, x_17 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_5),
             self.dwconv_w(x_11),
             self.dwconv_h(x_17)),
            dim=1,
        )


########################################

########################################
# ChannelAggregationFFN
def build_act_layer(act_type):
    # Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    # A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class ChannelAggregationFFN(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 mlp_ration=2.,
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * mlp_ration)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    # 输入 B C H W
    def forward(self, x):
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ChannelAggregationFFN
########################################

# Hybrid Inception-Style Large Kernel Convolutional Block renamed to CHB
class HybridInceptionBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            mlp_ratio: float = 2,
            split_ration=4,
            kernel_sizes=[0, 5, 11, 17]
    ):
        super().__init__()
        assert hidden_dim % split_ration == 0, "hidden_dim must be divisible by split_ration"

        self.hidden_dim = hidden_dim
        self.norm1 = nn.BatchNorm2d(hidden_dim)
        self.norm2 = nn.BatchNorm2d(hidden_dim)
        self.ISBlock = InceptionStyleDWConv2d(in_channels=hidden_dim, branch_ratio=split_ration,
                                              kernel_sizes=kernel_sizes)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(1e-1 * torch.ones(hidden_dim))
        self.CAFFN = ChannelAggregationFFN(embed_dims=hidden_dim, ffn_drop=drop_path, mlp_ration=mlp_ratio)
        self.CEB = LA(hidden_dim, mlp_ration=mlp_ratio)
        self.EGEM = GCE(hidden_dim, down_scale=8)

    def forward(self, input):
        # [b,c,h,w]
        x = self.norm1(input.contiguous())
        x = self.CEB(x).contiguous()

        convs = self.EGEM(x).contiguous()
        x_isb = self.ISBlock(x).contiguous()
        x_combined = x_isb + convs * self.skip_scale.view(1, self.hidden_dim, 1, 1)

        x_fused = self.CAFFN(self.norm2(x_combined))
        x_fused = x_fused.contiguous()

        x = x_fused + input
        return x


class BasicLayer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 kernel_sizes=[0, 5, 11, 17],
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(HybridInceptionBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                mlp_ratio=mlp_ratio,
                kernel_sizes=kernel_sizes
            ))

        self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


@ARCH_REGISTRY.register()
class EchoSR(nn.Module):
    def __init__(self,
                 in_chans=3,
                 embed_dim=180,
                 depths=(6, 6, 6, 6, 6, 6),
                 mlp_ratio=2.,
                 drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 drop_path_rate=0.1,
                 upsampler='pixelshuffle',
                 resi_connection='1conv',
                 kernel_sizes=[0, 5, 11, 17],
                 **kwargs):
        super(EchoSR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        print(f'start model\n')
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upsampler = upsampler
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depths))]  # stochastic depth decay rule

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ResidualGroup(
                dim=embed_dim,
                depth=depths[i_layer],
                mlp_ratio=self.mlp_ratio,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                kernel_sizes=kernel_sizes,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                resi_connection=resi_connection)
            self.layers.append(layer)
        # self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- restoration module ------------------------- #
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
        else:
            # for image denoising
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        # x_size = (x.shape[2], x.shape[3])

        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)

        else:
            # for image denoising
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for layer in self.layers:
            flops += layer.flops()
        flops += h * w * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


class UpsampleOneStep(nn.Sequential):
    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class ResidualGroup(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mlp_ratio=2.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 kernel_sizes=[0, 5, 11, 17],
                 resi_connection='1conv'):
        super(ResidualGroup, self).__init__()

        self.dim = dim
        self.residual_group = BasicLayer(
            dim=dim,
            depth=depth,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            kernel_sizes=kernel_sizes,
            use_checkpoint=use_checkpoint)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))
        self.conv = COFB(n_feats=dim)

    def forward(self, x):
        return self.conv(self.residual_group(x)) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def buildModel(upscale=4):
    return EchoSR(img_size=64,
                  patch_size=1,
                  in_chans=3,
                  embed_dim=60,
                  depths=(5, 5, 5, 5,),
                  kernel_sizes=[0, 5, 11, 17],
                  mlp_ratio=1.5,
                  drop_rate=0.1,
                  norm_layer=nn.LayerNorm,
                  patch_norm=True,
                  use_checkpoint=False,
                  upscale=upscale,
                  img_range=1.,
                  upsampler='pixelshuffledirect',
                  resi_connection='1conv')


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = buildModel(2).to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 512, 512)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 10

    # Warm up
    print('Warming up...\n')
    with torch.inference_mode():
        for _ in range(5):
            _ = model(input_data)

    # Reset CUDA memory stats
    torch.cuda.reset_peak_memory_stats()

    # Initialize CUDA events
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_inferences, 1))

    # Measure inference time
    print('Measuring inference time...\n')
    with torch.inference_mode():
        for i in tqdm.tqdm(range(num_inferences)):
            starter.record()
            output = model(input_data)
            ender.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            timings[i] = starter.elapsed_time(ender)  # Time in milliseconds

    average_inference_time = np.mean(timings)

    # Calculate memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # MB

    # Calculate parameters and MACs using thop
    macs, params = profile(model, inputs=(input_data,))

    # Output results
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average inference time over {num_inferences} runs: {average_inference_time:.4f} ms")
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.2f} MB")
    print(f"Number of parameters: {params / 1e3:.2f} K")
    print(f"MACs: {macs / 1e9:.2f} G")


if __name__ == '__main__':
    from thop import profile
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
