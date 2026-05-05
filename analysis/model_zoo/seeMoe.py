import time
from typing import Tuple, List

import numpy as np
import tqdm
from torch import Tensor, optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
import os
import psutil
# from basicsr.utils.registry import ARCH_REGISTRY


######################
# Meta Architecture
######################
# @ARCH_REGISTRY.register()
class SeemoRe(nn.Module):
    def __init__(self,
                 scale: int = 4,
                 in_chans: int = 3,
                 num_experts: int = 6,
                 num_layers: int = 6,
                 embedding_dim: int = 64,
                 img_range: float = 1.0,
                 use_shuffle: bool = False,
                 global_kernel_size: int = 11,
                 recursive: int = 2,
                 lr_space: int = 1,
                 topk: int = 2, ):
        super().__init__()
        self.scale = scale
        self.num_in_channels = in_chans
        self.num_out_channels = in_chans
        self.img_range = img_range

        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        # -- SHALLOW FEATURES --
        self.conv_1 = nn.Conv2d(self.num_in_channels, embedding_dim, kernel_size=3, padding=1)

        # -- DEEP FEATURES --
        self.body = nn.ModuleList(
            [ResGroup(in_ch=embedding_dim,
                      num_experts=num_experts,
                      use_shuffle=use_shuffle,
                      topk=topk,
                      lr_space=lr_space,
                      recursive=recursive,
                      global_kernel_size=global_kernel_size) for i in range(num_layers)]
        )

        # -- UPSCALE --
        self.norm = LayerNorm(embedding_dim, data_format='channels_first')
        self.conv_2 = nn.Conv2d(embedding_dim, embedding_dim, kernel_size=3, padding=1)
        self.upsampler = nn.Sequential(
            nn.Conv2d(embedding_dim, (scale ** 2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        # -- SHALLOW FEATURES --
        x = self.conv_1(x)
        res = x

        # -- DEEP FEATURES --
        for idx, layer in enumerate(self.body):
            x = layer(x)

        x = self.norm(x)

        # -- HR IMAGE RECONSTRUCTION --
        x = self.conv_2(x) + res
        x = self.upsampler(x)

        x = x / self.img_range + self.mean
        return x


#############################
# Components
#############################
class ResGroup(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 global_kernel_size: int = 11,
                 lr_space: int = 1,
                 topk: int = 2,
                 recursive: int = 2,
                 use_shuffle: bool = False):
        super().__init__()

        self.local_block = RME(in_ch=in_ch,
                               num_experts=num_experts,
                               use_shuffle=use_shuffle,
                               lr_space=lr_space,
                               topk=topk,
                               recursive=recursive)
        self.global_block = SME(in_ch=in_ch,
                                kernel_size=global_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.local_block(x)
        x = self.global_block(x)
        return x


#############################
# Global Block
#############################
class SME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int = 11):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = StripedConvFormer(in_ch=in_ch, kernel_size=kernel_size)

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


class StripedConvFormer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        self.to_qv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, padding=0),
            nn.GELU(),
        )

        self.attn = StripedConv2d(in_ch, kernel_size=kernel_size, depthwise=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, v = self.to_qv(x).chunk(2, dim=1)
        q = self.attn(q)
        x = self.proj(q * v)
        return x


#############################
# Local Blocks
#############################
class RME(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 lr_space: int = 1,
                 recursive: int = 2,
                 use_shuffle: bool = False, ):
        super().__init__()

        self.norm_1 = LayerNorm(in_ch, data_format='channels_first')
        self.block = MoEBlock(in_ch=in_ch, num_experts=num_experts, topk=topk, use_shuffle=use_shuffle,
                              recursive=recursive, lr_space=lr_space, )

        self.norm_2 = LayerNorm(in_ch, data_format='channels_first')
        self.ffn = GatedFFN(in_ch, mlp_ratio=2, kernel_size=3, act_layer=nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(self.norm_1(x)) + x
        x = self.ffn(self.norm_2(x)) + x
        return x


#################
# MoE Layer
#################
class MoEBlock(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int,
                 topk: int,
                 use_shuffle: bool = False,
                 lr_space: str = "linear",
                 recursive: int = 2):
        super().__init__()
        self.use_shuffle = use_shuffle
        self.recursive = recursive

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_ch, 2 * in_ch, kernel_size=1, padding=0)
        )

        self.agg_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=4, stride=4, groups=in_ch),
            nn.GELU())

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, groups=in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)
        )

        self.conv_2 = nn.Sequential(
            StripedConv2d(in_ch, kernel_size=3, depthwise=True),
            nn.GELU())

        if lr_space == "linear":
            grow_func = lambda i: i + 2
        elif lr_space == "exp":
            grow_func = lambda i: 2 ** (i + 1)
        elif lr_space == "double":
            grow_func = lambda i: 2 * i + 2
        else:
            raise NotImplementedError(f"lr_space {lr_space} not implemented")

        self.moe_layer = MoELayer(
            experts=[Expert(in_ch=in_ch, low_dim=grow_func(i)) for i in range(num_experts)],
            # add here multiple of 2 as low_dim
            gate=Router(in_ch=in_ch, num_experts=num_experts),
            num_expert=topk,
        )

        self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0)

    def calibrate(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        res = x

        for _ in range(self.recursive):
            x = self.agg_conv(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return res + x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)

        if self.use_shuffle:
            x = channel_shuffle(x, groups=2)
        x, k = torch.chunk(x, chunks=2, dim=1)

        x = self.conv_2(x)
        k = self.calibrate(k)

        x = self.moe_layer(x, k)
        x = self.proj(x)
        return x


class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 1):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert

    def forward(self, inputs: torch.Tensor, k: torch.Tensor):
        out = self.gate(inputs)
        weights = F.softmax(out, dim=1, dtype=torch.float).to(inputs.dtype)
        topk_weights, topk_experts = torch.topk(weights, self.num_expert)
        out = inputs.clone()

        if self.training:
            exp_weights = torch.zeros_like(weights)
            exp_weights.scatter_(1, topk_experts, weights.gather(1, topk_experts))
            for i, expert in enumerate(self.experts):
                out += expert(inputs, k) * exp_weights[:, i:i + 1, None, None]
        else:
            selected_experts = [self.experts[i] for i in topk_experts.squeeze(dim=0)]
            for i, expert in enumerate(selected_experts):
                out += expert(inputs, k) * topk_weights[:, i:i + 1, None, None]

        return out


class Expert(nn.Module):
    def __init__(self,
                 in_ch: int,
                 low_dim: int, ):
        super().__init__()
        self.conv_1 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_2 = nn.Conv2d(in_ch, low_dim, kernel_size=1, padding=0)
        self.conv_3 = nn.Conv2d(low_dim, in_ch, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.conv_2(k) * x  # here no more sigmoid
        x = self.conv_3(x)
        return x


class Router(nn.Module):
    def __init__(self,
                 in_ch: int,
                 num_experts: int):
        super().__init__()

        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(in_ch, num_experts, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


#################
# Utilities
#################
class StripedConv2d(nn.Module):
    def __init__(self,
                 in_ch: int,
                 kernel_size: int,
                 depthwise: bool = False):
        super().__init__()
        self.in_ch = in_ch
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=(1, self.kernel_size), padding=(0, self.padding),
                      groups=in_ch if depthwise else 1),
            nn.Conv2d(in_ch, in_ch, kernel_size=(self.kernel_size, 1), padding=(self.padding, 0),
                      groups=in_ch if depthwise else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def channel_shuffle(x, groups=2):
    bat_size, channels, w, h = x.shape
    group_c = channels // groups
    x = x.view(bat_size, groups, group_c, w, h)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(bat_size, -1, w, h)
    return x


class GatedFFN(nn.Module):
    def __init__(self,
                 in_ch,
                 mlp_ratio,
                 kernel_size,
                 act_layer, ):
        super().__init__()
        mlp_ch = in_ch * mlp_ratio

        self.fn_1 = nn.Sequential(
            nn.Conv2d(in_ch, mlp_ch, kernel_size=1, padding=0),
            act_layer,
        )
        self.fn_2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, padding=0),
            act_layer,
        )

        self.gate = nn.Conv2d(mlp_ch // 2, mlp_ch // 2,
                              kernel_size=kernel_size, padding=kernel_size // 2, groups=mlp_ch // 2)

    def feat_decompose(self, x):
        s = x - self.gate(x)
        x = x + self.sigma * s
        return x

    def forward(self, x: torch.Tensor):
        x = self.fn_1(x)
        x, gate = torch.chunk(x, 2, dim=1)

        gate = self.gate(gate)
        x = x * gate

        x = self.fn_2(x)
        return x


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def build(upscale=4):
    return SeemoRe(scale=upscale,
                   in_chans=3,
                   num_experts=3,
                   num_layers=16,
                   embedding_dim=48,
                   img_range=1.0,
                   use_shuffle=False,
                   global_kernel_size=11,
                   recursive=1,
                   lr_space="exp",
                   topk=1, )


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = build(4).to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 320, 180)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 1000

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
def simulate_training():
    # 设定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建模型实例
    model = build(4).to(device)

    # 设置损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 随机生成输入数据和目标数据
    input_shape = (32, 3, 64, 64)  # 假设批次大小为1
    input_data = torch.randn(input_shape).to(device)
    target = torch.randn(32,3,64*4,64*4).to(device)

    # 设定训练轮数
    num_epochs = 50
    total_time = 0

    # 模拟训练过程
    for epoch in range(num_epochs):
        start_time = time.time()

        # 前向传播
        output = model(input_data)

        # 计算损失
        loss = criterion(output, target)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 更新参数
        optimizer.step()

        end_time = time.time()
        total_time += (end_time - start_time)

        # 每轮结束后清理缓存
        torch.cuda.empty_cache()

    # 计算平均训练时间
    average_epoch_time = total_time / num_epochs

    # 同步操作
    torch.cuda.synchronize()

    # 计算显存占用
    memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # 单位MB
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # 单位MB
    max_memory_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)  # 单位MB

    # 使用 thop 库计算参数量和 MACs
    # macs, params = profile(model, inputs=(input_data,))

    # 输出结果
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average training time per epoch over {num_epochs} epochs: {average_epoch_time * 1000:.2f} mileseconds")
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.2f} MB")
    # print(f"Number of parameters: {params / 1e6:.2f} M")
    # print(f"MACs: {macs / 1e9:.2f} G")

# if __name__ == '__main__':
#     from thop import profile
#     import os

#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     main()
#     simulate_training()

def get_cpu_memory():
    """获取当前进程占用的物理内存 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def main():

    # 1. 强制设为 CPU 
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 建议：在移动端测试时，可以尝试限制线程数为 1 来测试单核性能极限，
    # 或者保持默认（全核）来模拟真实应用。
    # torch.set_num_threads(1) 

    # 2. 创建模型
    model = build(2).to(device).eval()

    # 3. 生成输入 (建议根据手机内存调整尺寸，512x512 对手机 CPU 压力很大)
    input_shape = (1, 3, 384, 384) 
    input_data = torch.randn(input_shape).to(device)

    # 4. 预热 (Warm up)
    # 移动端 CPU 必须要预热，否则 CPU 会从低频爬升，导致前几次数据极慢
    print('Warming up (2 runs)...')
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_data)

    # 5. 测量推理延迟
    num_inferences = 1
    print(f'Measuring inference time over {num_inferences} runs...')
    
    timings = []
    mem_before = get_cpu_memory()
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_inferences)):
            start_time = time.perf_counter()
            _ = model(input_data)
            end_time = time.perf_counter()
            
            # 记录毫秒
            timings.append((end_time - start_time) * 1000)

    mem_after = get_cpu_memory()
    avg_inference_time = np.mean(timings)
    std_inference_time = np.std(timings)

    # 6. 计算参数量和 MACs (thop 可以在 CPU 运行)
    try:
        from thop import profile
        macs, params = profile(model, inputs=(input_data,), verbose=False)
    except ImportError:
        print("thop not installed, skipping MACs calculation.")
        macs, params = 0, 0

    # 7. 输出结果
    print("\n" + "="*30)
    print(f"DEVICE: {device}")
    print(f"Input shape: {input_shape}")
    print(f"MACs: {macs / 1e9:.2f} G")
    print(f"Params: {params / 1e3:.2f} K")
    print("-" * 30)
    print(f"Avg Inference Time: {avg_inference_time:.2f} ms")
    print(f"Std Deviation: {std_inference_time:.2f} ms")
    print(f"Peak Mem Usage: ~{mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")
    print("="*30)

if __name__ == '__main__':
    main()