from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import os

import tqdm


def mean_channels_h(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True)
    return spatial_sum / F.size(3)


def stdv_channels_h(F):
    assert (F.dim() == 4)
    F_mean = mean_channels_h(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True) / F.size(3)
    return F_variance


def mean_channels_w(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(2, keepdim=True)
    return spatial_sum / F.size(2)


def stdv_channels_w(F):
    assert (F.dim() == 4)
    F_mean = mean_channels_w(F)
    F_variance = (F - F_mean).pow(2).sum(2, keepdim=True) / F.size(2)
    return F_variance


class DiVA_attention(nn.Module):
    def __init__(self):
        super(DiVA_attention, self).__init__()

        self.contrast_h = stdv_channels_h
        self.contrast_w = stdv_channels_w

        self.conv_h = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()

        c_h = self.contrast_h(x)
        c_w = self.contrast_w(x)

        a_h = self.conv_h(c_h).sigmoid()
        a_w = self.conv_w(c_w).sigmoid()

        out = identity * a_w * a_h

        return out


def init_weights(modules):
    pass


class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x


class UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, multi_scale, wn, group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, wn=wn, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, wn=wn, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, wn=wn, group=group)
        else:
            self.up = _UpsampleBlock(n_channels, scale=scale, wn=wn, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)


class _UpsampleBlock(nn.Module):
    def __init__(self, n_channels, scale, wn, group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []

        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [wn(nn.Conv2d(n_channels, 4 * n_channels, 3, 1, 1, groups=group)),
                            nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]

        elif scale == 3:
            modules += [wn(nn.Conv2d(n_channels, 9 * n_channels, 3, 1, 1, groups=group)), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        elif scale == 5:
            modules += [wn(nn.Conv2d(n_channels, 25 * n_channels, 3, 1, 1, groups=group)), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(5)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)

    def forward(self, x):
        out = self.body(x)
        return out


class BasicConv2d(nn.Module):

    def __init__(self, wn, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = wn(nn.Conv2d(in_planes, out_planes,
                                 kernel_size=kernel_size, stride=stride,
                                 padding=padding, bias=True))  # verify bias false

        self.LR = nn.ReLU(inplace=True)
        init_weights(self.modules)

    def forward(self, x):
        x = self.conv(x)
        x = self.LR(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 wn, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.DiVA = DiVA_attention()
        body = []
        expand = 6
        linear = 0.8
        body.append(
            wn(nn.Conv2d(64, 64 * expand, 1, padding=1 // 2)))
        body.append(nn.ReLU(inplace=True))
        body.append(
            wn(nn.Conv2d(64 * expand, int(64 * linear), 1, padding=1 // 2)))
        body.append(
            wn(nn.Conv2d(int(64 * linear), 64, 3, padding=3 // 2)))

        self.body = nn.Sequential(*body)

        init_weights(self.modules)

    def forward(self, x):
        out_x = self.body(x)
        out = out_x + x

        out_DiVA = self.DiVA(out_x)

        return out, out_DiVA


class RAFG(nn.Module):
    def __init__(self,
                 in_channels, out_channels, wn,
                 group=1):
        super(RAFG, self).__init__()

        self.rb1 = ResidualBlock(wn, 64, 64)
        self.rb2 = ResidualBlock(wn, 64, 64)
        self.rb3 = ResidualBlock(wn, 64, 64)

        self.reduction_1 = BasicConv2d(wn, 64 * 4, 64, 1, 1, 0)
        self.reduction_2 = BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1, A_1 = self.rb1(o0)

        b2, A_2 = self.rb2(b1)

        b3, A_3 = self.rb3(b2)

        Feature_bank = self.reduction_1(torch.cat([c0, b1, b2, b3], 1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))

        out = Feature_bank + x + Attention_bank

        return out, Attention_bank


class Net(nn.Module):

    def __init__(self, **kwargs):
        super(Net, self).__init__()

        wn = lambda x: torch.nn.utils.weight_norm(x)
        scale = 4
        group = 4

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        self.entry_1 = wn(nn.Conv2d(3, 64, 3, 1, 1))

        self.b1 = RAFG(64, 64, wn=wn)
        self.b2 = RAFG(64, 64, wn=wn)
        self.b3 = RAFG(64, 64, wn=wn)

        self.reduction_1 = BasicConv2d(wn, 64 * 4, 64, 1, 1, 0)
        self.reduction_2 = BasicConv2d(wn, 64 * 3, 64, 1, 1, 0)
        self.upsample = UpsampleBlock(64, scale=scale, multi_scale=False, wn=wn, group=group)

        self.exit1 = wn(nn.Conv2d(64, 3, 3, 1, 1))

    def forward(self, x, scale):
        x = self.sub_mean(x)
        res = x

        x = self.entry_1(x)
        c0 = o0 = x

        b1, A_1 = self.b1(o0)

        b2, A_2 = self.b2(b1)

        b3, A_3 = self.b3(b2)

        Feature_bank = self.reduction_1(torch.cat([c0, b1, b2, b3], 1))
        Attention_bank = self.reduction_2(torch.cat([A_1, A_2, A_3], 1))

        out = Feature_bank + x + Attention_bank

        out = self.upsample(out, scale=scale)

        out = self.exit1(out)

        skip = F.interpolate(res, (x.size(-2) * scale, x.size(-1) * scale), mode='bicubic', align_corners=False)

        out = skip + out

        out = self.add_mean(out)

        return out


def buildmodel(upscale=4):
    return Net()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = buildmodel().to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 320, 180)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 50

    # Warm up
    print('Warming up...\n')
    with torch.inference_mode():
        for _ in range(10):
            _ = model(input_data, 4)

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
            output = model(input_data, 4)
            ender.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded
            timings[i] = starter.elapsed_time(ender)  # Time in milliseconds

    average_inference_time = np.mean(timings)

    # Calculate memory usage
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # MB

    # Calculate parameters and MACs using thop
    macs, params = profile(model, inputs=(input_data,4))

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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
