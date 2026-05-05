import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import time

# ============================================================
# 1. 彻底替换 causal_conv1d
# ============================================================
def causal_conv1d_fn(x, weight, bias=None, activation=None):
    """
    原生 PyTorch 实现的因果卷积，兼容移动端 CPU
    x: (B, D, L)
    weight: (D, W)
    """
    B, D, L = x.shape
    width = weight.shape[-1]
    padding = width - 1
    
    # 使用分组卷积模拟原版逻辑
    # weight 为 (D, W), 需要转为 (D, 1, W) 用于 conv1d 的 groups=D
    x = F.conv1d(
        x, 
        weight.unsqueeze(1), 
        bias, 
        padding=padding, 
        groups=D
    )
    
    # 裁掉末尾多余的 padding 部分实现因果性
    out = x[..., :L]
    
    if activation == "silu" or activation == "swish":
        out = F.silu(out)
    return out

# ============================================================
# 2. 优化后的 selective_scan_ref
# ============================================================
def selective_scan_ref(
    u, delta, A, B, C,
    D=None, z=None,
    delta_bias=None,
    delta_softplus=False,
    return_last_state=False
):
    """
    纯 PyTorch 实现，移除了对 CUDA 的所有依赖。
    注意：在 CPU 上由于存在 Python 循环，序列长度 L 较大时会很慢。
    """
    dtype_in = u.dtype
    u = u.float()
    delta = delta.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)

    batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    # 预计算 deltaA 以减少循环内计算
    # 警告：如果 L 非常大，deltaA 可能会消耗大量内存
    # deltaA shape: (B, D, L, N)
    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    
    # 预计算 deltaB_u
    if not is_variable_B:
        deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
    else:
        if B.dim() == 3: # (B, N, L)
            deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
        else: # (B, G, N, L)
            B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
            deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)

    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])

    x = torch.zeros((batch, dim, dstate), device=u.device)
    ys = []

    # 核心循环
    # 提示：如果测试 512x512，请在测试脚本中将迭代次数设小，否则会跑很久
    for i in range(u.shape[2]):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if not is_variable_C:
            y = torch.einsum('bdn,dn->bd', x, C)
        else:
            if C.dim() == 3:
                y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
            else:
                y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
        ys.append(y)

    y = torch.stack(ys, dim=2)
    out = y if D is None else y + u * rearrange(D, "d -> d 1")
    if z is not None:
        out = out * F.silu(z)
    
    return (out.to(dtype=dtype_in), x) if return_last_state else out.to(dtype=dtype_in)

# ============================================================
# CPU-only Mamba inner (no CUDA)
# ============================================================

def mamba_inner_ref(
    xz,
    conv1d_weight,
    conv1d_bias,
    x_proj_weight,
    delta_proj_weight,
    out_proj_weight,
    out_proj_bias,
    A,
    B=None,
    C=None,
    D=None,
    delta_bias=None,
    B_proj_bias=None,
    C_proj_bias=None,
    delta_softplus=True
):
    L = xz.shape[-1]
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

    x, z = xz.chunk(2, dim=1)

    # Pure PyTorch causal conv
    x = causal_conv1d_fn(
        x,
        rearrange(conv1d_weight, "d 1 w -> d w"),
        conv1d_bias,
        "silu"
    )

    x_dbl = F.linear(
        rearrange(x, 'b d l -> (b l) d'),
        x_proj_weight
    )

    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=L)

    if B is None:
        B = x_dbl[:, delta_rank:delta_rank + d_state]
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            B = rearrange(
                B,
                "(b l) (dstate two) -> b dstate (l two)",
                l=L,
                two=2
            ).contiguous()

    if C is None:
        C = x_dbl[:, -d_state:]
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()
        else:
            C = rearrange(
                C,
                "(b l) (dstate two) -> b dstate (l two)",
                l=L,
                two=2
            ).contiguous()

    y = selective_scan_ref(
        x,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus
    )

    return F.linear(
        rearrange(y, "b d l -> b l d"),
        out_proj_weight,
        out_proj_bias
    )


# Alias
def mamba_inner_fn(
    *args,
    **kwargs
):
    return mamba_inner_ref(*args, **kwargs)


# 导出 selective_scan_fn
selective_scan_fn = selective_scan_ref

# 如果 MambaIR 还尝试从这里导入 causal_conv1d_fn
# 确保它也能找到
__all__ = ['selective_scan_fn', 'selective_scan_ref', 'mamba_inner_fn', 'causal_conv1d_fn']    