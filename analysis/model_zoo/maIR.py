# Code Implementation of the MaIR Model
import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable

import tqdm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import os
import psutil
import tqdm
from torch import optim
import sys
sys.path.append("/data/data/com.termux/files/home/storage/MyCode/SR/analysis/model_zoo")
# from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ops.mamba_ops_cpu import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import time
import sys
# from ops.mair_utils import mair_ids_generate, mair_ids_scan, mair_ids_inverse, \
#     mair_shift_ids_generate
# sys.path.append('/xlearning/boyun/codes/MaIR')
# try:
#     from ops.mair_utils import mair_ids_generate, mair_ids_scan, mair_ids_inverse, \
#         mair_shift_ids_generate
#     from basicsr.utils.registry import ARCH_REGISTRY
# except:
#     pass
import time
import torch
from einops import rearrange
import numpy as np
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def test_crop_by_horz(inp, scan_len):
    # Flip the return way
    split_inp = rearrange(inp, "h (d1 w) -> d1 h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    inp = rearrange(split_inp, " d1 h w -> (d1 h) w ")
    # print(inp)

    inp_window = rearrange(inp, "(d1 h) (d2 w) -> (d2 d1) h w ", h=2, w=scan_len)

    inp_window[:, -1] = inp_window[:, -1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(1, -1)
    print(inp_flatten)
    print(inp_flatten.shape)


def chw_2d(h, w):
    return torch.arange(1, (h * w + 1)).reshape(h, w)


def chw_3d(c, h, w):
    return torch.arange(1, (c * h * w + 1)).reshape(c, h, w)


def chw_4d(b, c, h, w, random=False):
    if random:
        return torch.randn(b * c * h * w).reshape(b, c, h, w)
    else:
        return torch.arange(1, (b * c * h * w + 1)).reshape(b, c, h, w)


def create_idx(b, c, h, w):
    # return torch.arange(1, (b*c*h*w+1)).reshape(b, c, h, w)
    return torch.arange(b * c * h * w).reshape(b, c, h, w)


def test_2d_horz(inp_h, inp_w):
    scan_len = 2
    # inp_h, inp_w = 4, 4
    # inp  = torch.randn((4,4))
    inp = torch.tensor([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]])
    inp = chw_2d(inp_h, inp_w)
    print(inp)
    test_crop_by_horz(inp, scan_len)


def sscan_einops(inp, scan_len):
    B, C, H, W = inp.shape
    # Flip the return way
    split_inp = rearrange(inp, "b c h (d1 w) -> d1 b c h w ", w=scan_len)
    for i in range(1, len(split_inp), 2):
        split_inp[i, :] = split_inp[i, :].flip(dims=[-2])
    reverse_inp = rearrange(split_inp, " d1 b c h w -> b c (d1 h) w ")
    # print(inp)

    inp_window = rearrange(reverse_inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)

    inp_window[:, -1] = inp_window[:, -1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)

    return inp_flatten


def sscan(inp, scan_len, shift_len=0):
    B, C, H, W = inp.shape
    # Flip the return way
    # 将返回的时候的列，上下翻转
    if shift_len == 0:
        for i in range(1, (W // scan_len) + 1, 2):
            # for j in range(scan_len):
            inp[:, :, :, i * scan_len:(i + 1) * scan_len] = inp[:, :, :, i * scan_len:(i + 1) * scan_len].flip(
                dims=[-2])
    else:
        for i in range(0, ((W - shift_len) // scan_len) + 1, 2):
            inp[:, :, :, (shift_len + i * scan_len):(shift_len + (i + 1) * scan_len)] = inp[:, :, :,
                                                                                        (shift_len + i * scan_len):(
                                                                                                    shift_len + (
                                                                                                        i + 1) * scan_len)].flip(
                dims=[-2])

    # 将当前return way的sub-line翻转
    # inp_window = rearrange(inp, "b c (d1 h) (d2 w) -> (b c d2 d1) h w ", h=2, w=scan_len)
    if shift_len == 0:
        for hi in range((H // 2)):
            for wi in range(W // scan_len):
                inp[:, :, 2 * hi + 1, wi * scan_len:(wi + 1) * scan_len] = inp[:, :, 2 * hi + 1,
                                                                           wi * scan_len:(wi + 1) * scan_len].flip(
                    dims=[-1])
    else:
        for hi in range((H // 2)):
            inp[:, :, 2 * hi + 1, 0:shift_len] = inp[:, :, 2 * hi + 1, 0:shift_len].flip(dims=[-1])

            for wi in range((W - shift_len) // scan_len):
                start_ = shift_len + wi * scan_len
                end_ = shift_len + (wi + 1) * scan_len
                inp[:, :, 2 * hi + 1, start_:end_] = inp[:, :, 2 * hi + 1, start_:end_].flip(dims=[-1])

    if (W - shift_len) % scan_len:
        # inp_last = inp[:,:,:,-(W % scan_len):].reshape(B, C, -1)
        inp_last = inp[:, :, :, -((W - shift_len) % scan_len):]
        inp_last[:, :, 1::2, :] = inp_last[:, :, 1::2, :].flip(dims=[-1])  # 取偶数位，奇数位是::2
        inp_last = inp_last.reshape(B, C, -1)

        inp_rest = inp[:, :, :, :-((W - shift_len) % scan_len)]
    else:
        inp_rest = inp

    if shift_len == 0:
        inp_window = rearrange(inp_rest, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)
    else:
        inp_first = inp_rest[:, :, :, :shift_len].reshape(B, C, -1)

        inp_middle = inp_rest[:, :, :, shift_len:]
        inp_window = rearrange(inp_middle, "b c h (d2 w) -> (b c d2) h w ", w=scan_len)

    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    inp_flatten = inp_window.reshape(B, C, -1)
    # inp_window[:,-1] = inp_window[:,-1].flip(dims=[-1])
    # inp_flatten = inp.reshape(B, C, 1, -1)
    # print(inp_flatten)
    # print(inp_flatten.shape)
    if shift_len != 0:
        inp_flatten = torch.concat((inp_first, inp_flatten), dim=-1)

    if (W - shift_len) % scan_len:
        inp_flatten = torch.concat((inp_flatten, inp_last), dim=-1)
        # print(inp_last.shape)
    return inp_flatten


# def sscan_4d(inp, scan_len, ues_einops=True, fix_ending=True):
def sscan_4d(inp, scan_len, shift_len=0, fix_ending=True, use_einops=False):
    B, C, H, W = inp.shape
    L = H * W
    if fix_ending:
        inp_reverse = torch.flip(inp, dims=[-1, -2])
        inp_cat = torch.concat((inp, inp_reverse), dim=1)
        inp_cat_t = inp_cat.transpose(-1, -2).contiguous()

        if use_einops:
            line1 = sscan_einops(inp_cat, scan_len)
            line2 = sscan_einops(inp_cat_t, scan_len)
        else:
            line1 = sscan(inp_cat, scan_len, shift_len)
            line2 = sscan(inp_cat_t, scan_len, shift_len)

        xs = torch.stack([line1.reshape(B, 2, -1, L), line2.reshape(B, 2, -1, L)], dim=1).reshape(B, 4, -1, L)
    else:
        inp_t = inp.transpose(-1, -2).contiguous()
        if use_einops:
            line1 = sscan_einops(inp, scan_len)
            line2 = sscan_einops(inp_t, scan_len)
        else:
            line1 = sscan(inp, scan_len, shift_len)
            line2 = sscan(inp_t, scan_len, shift_len)

        x_hwwh = torch.stack([line1.reshape(B, -1, L), line2.reshape(B, -1, L)], dim=1).reshape(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        # print(xs)
    return xs


def inverse_ids_generate(origin_ids, K=4):
    '''
        Input: origin_ids: (B, K, C, L)
        Output: (B, K, C, L)
        Note: C is set to 1 for speeding up.
    '''
    inverse_ids = torch.argsort(origin_ids, dim=-1)
    return inverse_ids


def mair_ids_generate(inp_shape, scan_len=4, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # inp_idx = create_idx(1, inp_c, inp_h, inp_w)
    inp_idx = create_idx(1, 1, inp_h, inp_w)

    xs_scan_ids = sscan_4d(inp_idx, scan_len)[0]

    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)

    return xs_scan_ids, xs_inverse_ids


def mair_shift_ids_generate(inp_shape, scan_len=4, shift_len=0, K=4):
    inp_b, inp_c, inp_h, inp_w = inp_shape

    # create_idx函数运行时间：0.0050699710845947266 秒
    # start_time = time.time()
    inp_idx = create_idx(1, 1, inp_h, inp_w)
    # print(f"create_idx函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_scan_ids = sscan_4d(inp_idx, scan_len, shift_len=shift_len)[0]
    # print(f"sscan_4d函数运行时间：{time.time() - start_time} 秒")

    # xs_scan_ids函数运行时间：0.05201005935668945 秒
    # start_time = time.time()
    xs_scan_ids = xs_scan_ids.repeat(inp_b, 1, 1, 1)
    # print(f"xs_scan_ids函数运行时间：{time.time() - start_time} 秒")

    # start_time = time.time()
    xs_inverse_ids = inverse_ids_generate(xs_scan_ids, K=K)
    # print(f"inverse_ids_generate函数运行时间：{time.time() - start_time} 秒")

    return xs_scan_ids, xs_inverse_ids


def mair_ids_scan(inp, xs_scan_ids, bkdl=False, K=4):
    '''
        inp: B, C, H, W
        xs_scan_ids: K, 1, L
    '''
    B, C, H, W = inp.shape
    L = H * W
    xs_scan_ids = xs_scan_ids.reshape(K, L)

    y1 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[0])
    y2 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[1])
    y3 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[2])
    y4 = torch.index_select(inp.reshape(B, 1, C, -1), -1, xs_scan_ids[3])

    if bkdl:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1)
    else:
        inp_flatten = torch.cat((y1, y2, y3, y4), dim=1).reshape(B, 4, -1)
    return inp_flatten


def mair_ids_inverse(inp, xs_scan_ids, shape=None):
    '''
        inp: (B, K, -1, L)
        xs_scan_ids: (1, K, 1, L)
    '''
    B, K, _, L = inp.shape
    xs_scan_ids = xs_scan_ids.reshape(K, L)
    if not shape:
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, L)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, L)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, L)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, L)
    else:
        B, C, H, W = shape
        y1 = torch.index_select(inp[:, 0, :], -1, xs_scan_ids[0]).reshape(B, -1, H, W)
        y2 = torch.index_select(inp[:, 1, :], -1, xs_scan_ids[1]).reshape(B, -1, H, W)
        y3 = torch.index_select(inp[:, 2, :], -1, xs_scan_ids[2]).reshape(B, -1, H, W)
        y4 = torch.index_select(inp[:, 3, :], -1, xs_scan_ids[3]).reshape(B, -1, H, W)
    return torch.cat((y1, y2, y3, y4), dim=1)


def test_time():
    scan_len = 4
    shift_len = 2
    inp_b, inp_c, inp_h, inp_w = 2, 3, 3, 4
    inp = chw_4d(1, 1, inp_h, inp_w, random=False)
    inp_rgb = chw_4d(inp_b, inp_c, inp_h, inp_w, random=False)
    print("inp:", inp_rgb)

    # Original
    xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp.shape, scan_len=scan_len, K=4)
    xs = mair_ids_scan(inp_rgb, xs_scan_ids, bkdl=True)
    inp_flatten = mair_ids_inverse(xs, xs_inverse_ids, shape=(inp_b, inp_c, inp_h, inp_w))

    inp_flatten = inp_flatten.chunk(4, dim=1)
    print("recovered input:")
    for i in range(len(inp_flatten)):
        print("inp_flatten:", i)
        print(inp_flatten[i])
    print("end")


# if __name__ == '__main__':
#     # torch.set_default_device(1)
#     start_time = time.time()
#     result = test_time()
#     end_time = time.time()
#
#     print(f"函数运行时间：{end_time - start_time} 秒")

NEG_INF = -1000000


class ShuffleAttn(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, group=4, act_layer=nn.GELU,
                 input_resolution=(64, 64)):
        super().__init__()
        self.group = group
        self.input_resolution = input_resolution
        self.in_features = in_features
        self.out_features = out_features

        self.gating = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def channel_rearrange(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, self.group, group_channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def forward(self, x):
        x = self.channel_shuffle(x)
        x = self.gating(x)
        x = self.channel_rearrange(x)

        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        # nn.AdaptiveAvgPool2d(1),
        flops += H * W * self.in_features

        # nn.Conv2d(in_features, out_features, groups=self.group, kernel_size=1, stride=1, padding=0),
        flops += H * W * self.in_features * self.out_features // self.group

        # nn.Sigmoid()
        flops += H * W * self.out_features * 4
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 input_resolution=(64, 64)):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.input_resolution = input_resolution
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution

        flops += 2 * H * W * self.in_features * self.hidden_features
        flops += H * W * self.hidden_features

        return flops


class VMM(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            input_resolution=(64, 64),
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.input_resolution = input_resolution

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self.gating = ShuffleAttn(in_features=self.d_inner * 4, out_features=self.d_inner * 4, group=self.d_inner)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor,
                     mair_ids,
                     x_proj_bias: torch.Tensor = None,
                     ):
        # print(x.shape) C=360
        B, C, H, W = x.shape
        L = H * W
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        K = 4
        # print("hello")
        xs = mair_ids_scan(x, mair_ids[0])

        x_dbl = F.conv1d(xs.reshape(B, -1, L), self.x_proj_weight.reshape(-1, D, 1),
                         bias=(x_proj_bias.reshape(-1) if x_proj_bias is not None else None), groups=K)
        dts, Bs, Cs = torch.split(x_dbl.reshape(B, K, -1, L), [R, N, N], dim=2)
        dts = F.conv1d(dts.reshape(B, -1, L), self.dt_projs_weight.reshape(K * D, -1, 1), groups=K)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        out_y = self.selective_scan(
            xs, dts,
            -torch.exp(self.A_logs.float()).view(-1, self.d_state), Bs, Cs, self.Ds.float().view(-1), z=None,
            delta_bias=self.dt_projs_bias.float().view(-1),
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return mair_ids_inverse(out_y, mair_ids[1], shape=(B, -1, H, W))  # B, C, L

    def forward(self, x: torch.Tensor, mair_ids, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x, mair_ids)
        assert y.dtype == torch.float32
        y = y * self.gating(y)
        y1, y2, y3, y4 = torch.chunk(y, 4, dim=1)
        y = y1 + y2 + y3 + y4
        y = y.permute(0, 2, 3, 1).contiguous()

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        if self.dropout is not None:
            y = self.dropout()
        return y

    def flops_forward_core(self, H, W):
        flops = 0
        # flops of x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) in Core
        flops += 4 * (H * W) * self.d_inner * (self.dt_rank + self.d_state * 2)
        # flops of dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dt_rank=12, d_inner=360
        flops += 4 * (H * W) * self.dt_rank * self.d_inner
        # print(flops/1e6, (4 * H * W) * (self.d_state * self.d_state * 2)/1e6)
        # 610.46784 M 8.388608 M

        # Flops of discretization
        flops += (4 * H * W) * (self.d_state * self.d_state * 2)

        # Flops of Vmamba selective_scan
        # # h' = Ah(t) + Bx(t)
        # flops += (4 * H * W) * (self.d_state * self.d_state + self.d_inner * self.d_state)
        # # y = Ch(t) + DBx(t)
        # flops += (4 * H * W) * (self.d_inner * self.d_inner + self.d_inner * self.d_state)
        # 640*360*36*90*16/1e9=11.94G
        flops += 4 * 9 * H * W * self.d_inner * self.d_state
        # print(4 * 9 * H * W * self.d_inner * self.d_state/1e9)
        return flops

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flop of in_proj
        flops += H * W * self.d_model * self.d_inner * 2
        # flops of x = self.act(self.conv2d(x))
        flops += H * W * self.d_inner * 3 * 3 + H * W * self.d_inner
        # print(H, W, self.d_state, self.d_inner)
        flops += self.flops_forward_core(H, W)
        # 64 64 16 360
        flops += self.gating.flops()
        # y = y1 + y2 + y3 + y4
        flops += 4 * H * W * self.d_inner
        # flops of y = self.out_norm(y)
        flops += H * W * self.d_inner
        # flops of y = y * F.silu(z)
        flops += 2 * H * W * self.d_inner

        # flops of out = self.out_proj(y)
        flops += H * W * self.d_inner * self.d_model

        return flops


class RMB(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            ssm_ratio: float = 2.,
            input_resolution=(64, 64),
            is_light_sr: bool = False,
            shift_size=0,
            mlp_ratio=1.5,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = VMM(d_model=hidden_dim, d_state=d_state, expand=ssm_ratio, dropout=attn_drop_rate,
                                  input_resolution=input_resolution, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.conv_blk = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, input_resolution=input_resolution)

        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))
        self.hidden_dim = hidden_dim
        self.input_resolution = input_resolution

        self.shift_size = shift_size

    def forward(self, input, mair_ids, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]

        x = self.ln_1(input)
        if self.shift_size > 0:
            x = input * self.skip_scale + self.drop_path(self.self_attention(x, (mair_ids[2], mair_ids[3])))
        else:
            x = input * self.skip_scale + self.drop_path(self.self_attention(x, (mair_ids[0], mair_ids[1])))

        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x))

        x = x.reshape(B, -1, C)
        return x

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # flops of norm1 self.ln_1 -> layer_norm1
        flops += self.hidden_dim * H * W
        # flops of SS2D
        flops += self.self_attention.flops()
        # flops of input * self.skip_scale and residual
        flops += self.hidden_dim * H * W * 2
        # flops of norm2 self.ln_2 -> layer_norm2
        flops += self.hidden_dim * H * W
        # flops of MLP
        flops += self.conv_blk.flops()
        # flops of input * self.skip_scale2 and residual
        flops += self.hidden_dim * H * W * 2

        return flops


class BasicLayer(nn.Module):
    """ The Basic MaIR Layer in one Residual Mamba Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 ssm_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 scan_len=4,
                 mlp_ratio=2
                 ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.ssm_ratio = ssm_ratio
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(RMB(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                ssm_ratio=self.ssm_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                shift_size=0 if (i % 2 == 0) else scan_len // 2,
                mlp_ratio=self.mlp_ratio)
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, mair_ids, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mair_ids, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


# @ARCH_REGISTRY.register()
class MaIR(nn.Module):
    r""" Mamba-based Image Restoration Network (MaIR)
           A PyTorch implementation of : `MaIR: A Locality- and Continuity-Preserving Mamba for Image Restoration`.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           ssm_ratio (int): enlarge ratio in MaIR Module
           mlp_ratio (int): enlarge ratio in the hidden space of MLP
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
           scan_len: Stripe width of the NSS
       """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=60,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=16,
                 ssm_ratio=1.5,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv',
                 dynamic_ids=False,
                 scan_len=8,
                 mlp_ratio=2,
                 **kwargs):

        super(MaIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.ssm_ratio = ssm_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.num_out_ch = num_out_ch

        self.dynamic_ids = dynamic_ids
        self.scan_len = scan_len
        img_size_ids = to_2tuple(img_size)
        self.image_size = img_size_ids

        if not self.dynamic_ids:
            self._generate_ids((1, 1, img_size_ids[0], img_size_ids[1]))

        # transfer 2D feature map into 1D token sequence, pay attention to whether using normalization
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == 'pixelshuffledirect' else False
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build Residual State Space Group (RSSG)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):  # 6-layer
            layer = RMG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                d_state=d_state,
                ssm_ratio=self.ssm_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  # no impact on SR results
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
                is_light_sr=self.is_light_sr,
                scan_len=scan_len,
                mlp_ratio=mlp_ratio
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in the end of all residual groups
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------3. high-quality image reconstruction ------------------------ #
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

    def _generate_ids(self, inp_shape):
        B, C, H, W = inp_shape

        xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)  # [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_scan_ids = xs_scan_ids.cuda()
            self.xs_inverse_ids = xs_inverse_ids.cuda()
        else:
            self.xs_scan_ids = xs_scan_ids
            self.xs_inverse_ids = xs_inverse_ids

        xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W),
                                                                          scan_len=self.scan_len,
                                                                          shift_len=self.scan_len // 2)  # [B,H,W,C]
        if torch.cuda.is_available():
            self.xs_shift_scan_ids = xs_shift_scan_ids.cuda()
            self.xs_shift_inverse_ids = xs_shift_inverse_ids.cuda()
        else:
            self.xs_shift_scan_ids = xs_shift_scan_ids
            self.xs_shift_inverse_ids = xs_shift_inverse_ids

        del xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids

    def forward_features(self, x):
        B, C, H, W = x.shape
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)  # N,L,C
        x = self.pos_drop(x)

        if self.dynamic_ids or (self.image_size != (H, W)):
            xs_scan_ids, xs_inverse_ids = mair_ids_generate(inp_shape=(1, 1, H, W), scan_len=self.scan_len)  # [B,H,W,C]
            xs_shift_scan_ids, xs_shift_inverse_ids = mair_shift_ids_generate(inp_shape=(1, 1, H, W),
                                                                              scan_len=self.scan_len,
                                                                              shift_len=self.scan_len // 2)  # [B,H,W,C]
            if torch.cuda.is_available():
                xs_scan_ids, xs_inverse_ids = xs_scan_ids.cuda(), xs_inverse_ids.cuda()
                xs_shift_scan_ids, xs_shift_inverse_ids = xs_shift_scan_ids.cuda(), xs_shift_inverse_ids.cuda()
            for layer in self.layers:
                x = layer(x, (xs_scan_ids, xs_inverse_ids, xs_shift_scan_ids, xs_shift_inverse_ids), x_size)
        else:
            for layer in self.layers:
                x = layer(x, (self.xs_scan_ids, self.xs_inverse_ids, self.xs_shift_scan_ids, self.xs_shift_inverse_ids),
                          x_size)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

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

    def flops_layers(self):
        flops = 0
        h, w = self.patches_resolution

        # flops of forward_features
        flops += self.patch_embed.flops()
        print("self.patches_resolution:", self.patches_resolution)

        for layer in self.layers:
            flops += layer.flops()

        # flops of self.norm
        flops += h * w * self.embed_dim

        # flops of self.patch_unembed
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of self.conv_after_body
        flops += h * w * 9 * self.embed_dim * self.embed_dim

        # flops of Residual
        flops += h * w * self.embed_dim

        return flops

    def flops(self):
        flops = 0
        h, w = self.patches_resolution
        # x = self.conv_first(x)
        flops += h * w * 3 * self.embed_dim * 9

        if self.upsampler == 'pixelshuffle':
            # for classical SR

            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # x = self.conv_before_upsample(x)
            # nn.Conv2d(embed_dim, num_feat (=64), 3, 1, 1), nn.LeakyReLU(inplace=True))
            flops += h * w * 9 * self.embed_dim * 64
            flops += h * w * 64

            # self.upsample(x)
            if self.upscale == 2:
                flops += h * w * 9 * 64 * 4 * 64
            elif self.upscale == 3:
                flops += h * w * 9 * 64 * 9 * 64
            # x = self.conv_last()
            flops += h * w * 9 * 64 * 3

        elif self.upsampler == 'pixelshuffledirect':
            # x = self.conv_after_body(self.forward_features(x)) + x
            flops += self.flops_layers()

            # flops of UpsampleOneStep
            # self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch)
            flops += h * w * 9 * self.embed_dim * (self.upscale ** 2) * self.num_out_ch

        return flops


class RMG(nn.Module):
    """Residual Mamba Group (RMG).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 d_state=16,
                 ssm_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=None,
                 patch_size=None,
                 resi_connection='1conv',
                 is_light_sr=False,
                 scan_len=4,
                 mlp_ratio=2
                 ):
        super(RMG, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution  # [64, 64]

        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            is_light_sr=is_light_sr,
            scan_len=scan_len,
            mlp_ratio=mlp_ratio
        )

        # build the last conv layer in each residual state space group
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

    def forward(self, x, mair_ids, x_size):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, mair_ids, x_size), x_size))) + x

    def flops(self):
        flops = 0
        flops += self.residual_group.flops()
        h, w = self.input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    r""" transfer 2D feature map into 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        h, w = self.img_size
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" return 2D feature map from 1D token sequence

    Args:
        img_size (int): Image size.  Default: None.
        patch_size (int): Patch token size. Default: None.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self):
        flops = 0
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

# tiny version
def buildMaIR(upscale=4):
    return MaIR(img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=60,
                 depths=(6, 6, 6, 6),
                 drop_rate=0.,
                 d_state=1,
                 ssm_ratio=1.1,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=upscale,
                 img_range=1.,
                 upsampler='pixelshuffledirect',
                 resi_connection='1conv',
                 dynamic_ids=False,
                 scan_len=8,
                 mlp_ratio=1.6)


def main():

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create model instance
    model = buildMaIR(4).to(device).eval()

    # Generate random input data
    input_shape = (1, 3, 204, 204)  # Batch size of 1
    input_data = torch.randn(input_shape).to(device)

    # Set number of inferences
    num_inferences = 10

    # Warm up
    print('Warming up...\n')
    with torch.inference_mode():
        for _ in range(2):
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
    # 使用 fvcore 计算 FLOPs 和 参数量
    # flops2 = FlopCountAnalysis(model, input_data)
    # params2 = parameter_count(model)
    # Output results
    print(f"Input shape: {input_shape}")
    print(f"Output shape: {output.shape}")
    print(f"Average inference time over {num_inferences} runs: {average_inference_time:.4f} ms")
    print(f"Memory allocated: {memory_allocated:.2f} MB")
    print(f"Max memory allocated: {max_memory_allocated:.2f} MB")
    print(f"Max memory reserved: {max_memory_reserved:.2f} MB")
    print(f"Number of parameters: {params / 1e3:.2f} K")
    # print(f"Number of parameters: {params2[''] / 1e3:.2f} K")
    print(f"MACs: {macs / 1e9:.2f} G")
    # print(f"MACs: {flops2.total() / 1e9:.2f} G")




# if __name__ == '__main__':
#     from thop import profile
#     # mamba based used
#     from fvcore.nn import FlopCountAnalysis, parameter_count
#     import os

#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#     main()

def get_cpu_memory():
    """获取当前进程占用的物理内存 (MB)"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)

def main():
    from fvcore.nn import FlopCountAnalysis, parameter_count
    # 1. 强制设为 CPU 
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 建议：在移动端测试时，可以尝试限制线程数为 1 来测试单核性能极限，
    # 或者保持默认（全核）来模拟真实应用。
    # torch.set_num_threads(1) 

    # 2. 创建模型
    model = buildMaIR(2).to(device).eval()

    # 3. 生成输入 (建议根据手机内存调整尺寸，512x512 对手机 CPU 压力很大)
    input_shape = (1, 3, 128, 128) 
    input_data = torch.randn(input_shape).to(device)

    # 4. 预热 (Warm up)
    # 移动端 CPU 必须要预热，否则 CPU 会从低频爬升，导致前几次数据极慢
    print('Warming up (2 runs)...')
    with torch.no_grad():
        for _ in range(2):
            _ = model(input_data)

    # 5. 测量推理延迟
    num_inferences = 5
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
        # 使用 fvcore 计算 FLOPs 和 参数量
        # flops = FlopCountAnalysis(model, input_data)
        # params = parameter_count(model)
    except ImportError:
        print("thop not installed, skipping MACs calculation.")
        macs, params = 0, 0

    # 7. 输出结果
    print("\n" + "="*30)
    print(f"DEVICE: {device}")
    print(f"Input shape: {input_shape}")
    print(f"MACs: {macs / 1e9:.2f} G")
    # print(f"Flops: {flops.total() / 1e9:.2f} G")
    print(f"Params: {params / 1e3:.2f} K")
    print("-" * 30)
    print(f"Avg Inference Time: {avg_inference_time:.2f} ms")
    print(f"Std Deviation: {std_inference_time:.2f} ms")
    print(f"Peak Mem Usage: ~{mem_after:.2f} MB (Delta: {mem_after - mem_before:.2f} MB)")
    print("="*30)

if __name__ == '__main__':
    main()