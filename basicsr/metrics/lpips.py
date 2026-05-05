import cv2
import numpy as np
import torch
import pyiqa
from pyiqa import create_metric

from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

metric_niqe = create_metric('niqe', device=torch.device(device))


@METRIC_REGISTRY.register()
def calculate_niqe(img, img2, crop_border, input_order='HWC', test_y_channel=False, device='cpu'):
    """Calculate NIQE (Natural Image Quality Evaluator).

    Note:
        NIQE is a no-reference metric, img2 is ignored.

    Args:
        img (ndarray): Image with range [0, 255].
        img2 (ndarray): Not used, kept for interface consistency.
        crop_border (int): Cropped pixels in each edge.
        input_order (str): 'HWC' or 'CHW'.
        test_y_channel (bool): Whether to use Y channel.
        device (str): 'cpu' or 'cuda'.

    Returns:
        float: NIQE score (lower is better).
    """

    # 检查输入顺序
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}.')

    # 调整顺序
    img = reorder_image(img, input_order=input_order)

    # 转 float
    img = img.astype(np.float32)

    # crop
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]

    # Y通道（NIQE通常用灰度）
    if test_y_channel:
        img = to_y_channel(img)

    # ⚠️ pyiqa 的 NIQE 推荐输入范围 [0,1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    img_tensor = img_tensor.to(device)

    # 创建 metric（建议放外面缓存，但先这样写）

    score = metric_niqe(img_tensor)

    return score.item()

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, device='cpu'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        device (str): Device to use for evaluation ('cpu' or 'cuda'). Default: 'cpu'.

    Returns:
        float: lpips result.
    """

    # 确认输入图像形状一致
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # 检查输入顺序是否正确
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    # 调整图像顺序
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # 将图像转换为浮点数
    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 如果有裁剪边界，则去除边缘像素
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 如果测试Y通道，则将图像转换为YCbCr格式并只保留Y通道
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    # 创建LPIPS度量对象
    metric_lpips = create_metric('lpips-vgg', device=torch.device(device))

    # 将numpy数组转换为torch张量，并调整维度以适应模型输入
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.0

    # 移动到指定设备
    img_tensor = img_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # 计算LPIPS分数
    score = metric_lpips(img_tensor, img2_tensor)

    return score.item()

@METRIC_REGISTRY.register()
def calculate_musiq(img, img2, crop_border, input_order='HWC', test_y_channel=False, device='cpu'):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.
        device (str): Device to use for evaluation ('cpu' or 'cuda'). Default: 'cpu'.

    Returns:
        float: lpips result.
    """

    # 确认输入图像形状一致
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    # 检查输入顺序是否正确
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')

    # 调整图像顺序
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    # 将图像转换为浮点数
    img = img.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 如果有裁剪边界，则去除边缘像素
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 如果测试Y通道，则将图像转换为YCbCr格式并只保留Y通道
    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    # 创建LPIPS度量对象
    metric_musiq = create_metric('musiq', device=torch.device(device))

    # 将numpy数组转换为torch张量，并调整维度以适应模型输入
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0) / 255.0

    # 移动到指定设备
    img_tensor = img_tensor.to(device)
    img2_tensor = img2_tensor.to(device)

    # 计算LPIPS分数
    score = metric_musiq(img_tensor)

    return score.item()