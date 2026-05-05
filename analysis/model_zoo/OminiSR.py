#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: OmniSR.py
# Created Date: Tuesday April 28th 2022
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 23rd April 2023 3:06:36 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################
import numpy as np
import torch
import torch.nn as nn
import tqdm
from ops.OSAG import OSAG
from ops.pixelshuffle import pixelshuffle_block
import torch.nn.functional as F


class OmniSR(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, **kwargs):
        super(OmniSR, self).__init__()

        res_num = 5
        up_scale = 4
        bias = False

        residual_layer = []
        self.res_num = res_num

        for _ in range(res_num):
            temp_res = OSAG(channel_num=num_feat, **kwargs)
            residual_layer.append(temp_res)
        self.residual_layer = nn.Sequential(*residual_layer)
        self.input = nn.Conv2d(in_channels=num_in_ch, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                               bias=bias)
        self.output = nn.Conv2d(in_channels=num_feat, out_channels=num_feat, kernel_size=3, stride=1, padding=1,
                                bias=bias)
        self.up = pixelshuffle_block(num_feat, num_out_ch, up_scale, bias=bias)

        # self.tail   = pixelshuffle_block(num_feat,num_out_ch,up_scale,bias=bias)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, sqrt(2. / n))

        self.window_size = 8
        self.up_scale = up_scale

    def check_image_size(self, x):
        _, _, h, w = x.size()
        # import pdb; pdb.set_trace()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        # x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'constant', 0)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        residual = self.input(x)
        out = self.residual_layer(residual)

        # origin
        out = torch.add(self.output(out), residual)
        out = self.up(out)

        out = out[:, :, :H * self.up_scale, :W * self.up_scale]
        return out

def buildModel(upscale=4):
    return OmniSR()


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = buildModel().to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 320, 180)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 200

    # Warm up
    print('Warming up...\n')
    with torch.inference_mode():
        for _ in range(10):
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