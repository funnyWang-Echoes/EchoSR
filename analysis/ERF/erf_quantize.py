#
#
# from model_zoo.mambaIR import buildMambaIR
from analysis.model_zoo.MAN import buildMan
from model_zoo.myModel import buildModel
import os
import time
from functools import partial
from typing import Callable
import seaborn
from model_zoo.swinIR import buildSwinIR
from model_zoo.rcan import buildRCAN
from model_zoo.edsr import buildEDSR
from model_zoo.hat import HAT
import numpy as np
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from basicsr.data.data_util import paired_paths_from_folder, paired_paths_from_lmdb, paired_paths_from_meta_info_file
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
import torch
import torch.nn as nn
from torch import optim as optim
from torchvision import datasets, transforms
from timm.utils import AverageMeter
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from basicsr.utils.options import dict2str, parse_options

# 配置文件解析
root_path = r'F:\code\CodeProject\HLKNet\options\HLKNet_test\test_HLKNet_light_SRx4.yml'
opt, _ = parse_options(root_path, is_train=False)
opt = opt['datasets']['test_1']  # we use the 4-th SR testsets(i.e. Urban100) to visualize ERF.


# ==========================
# 1. 数据加载与预处理
# ==========================
class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.io_backend_type = self.io_backend_opt.get('type', 'disk')
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.task = opt['task'] if 'task' in opt else None
        self.noise = opt['noise'] if 'noise' in opt else 0
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        if self.io_backend_type == 'lmdb':
            self.io_backend_opt['db_paths'] = [self.lq_folder, self.gt_folder]
            self.io_backend_opt['client_keys'] = ['lq', 'gt']
            self.paths = paired_paths_from_lmdb([self.lq_folder, self.gt_folder], ['lq', 'gt'])
        elif 'meta_info_file' in self.opt and self.opt['meta_info_file'] is not None:
            self.paths = paired_paths_from_meta_info_file([self.lq_folder, self.gt_folder], ['lq', 'gt'],
                                                          self.opt['meta_info_file'], self.filename_tmpl)
        else:
            self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder], ['lq', 'gt'], self.filename_tmpl,
                                                  self.task)

    def __getitem__(self, index):
        if self.file_client is None:
            temp_io_args = self.io_backend_opt.copy()
            backend_type = temp_io_args.pop('type', self.io_backend_type)
            self.file_client = FileClient(backend_type, **temp_io_args)

        scale = self.opt['scale']

        if self.task == 'CAR':
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.
        elif self.task == 'denoising_gray':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)
            img_gt = np.expand_dims(img_gt, axis=2)
            img_lq = np.expand_dims(img_lq, axis=2)
        elif self.task == 'denoising_color':
            gt_path = self.paths[index]['gt_path']
            lq_path = gt_path
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            if self.opt['phase'] != 'train':
                np.random.seed(seed=0)
            img_lq = img_gt + np.random.normal(0, self.noise / 255., img_gt.shape)
        else:
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)


# ==========================
# 2. 基础可视化工具
# ==========================
def setup_plot_style():
    plt.rcParams["font.family"] = "Times New Roman"
    large, med, small = 24, 24, 24
    params = {'axes.titlesize': large, 'legend.fontsize': med, 'figure.figsize': (16, 10),
              'axes.labelsize': med, 'xtick.labelsize': med, 'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    plt.rcParams['axes.unicode_minus'] = False


setup_plot_style()


def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25), norm_val=None):
    def heatmap(data, camp='RdYlGn', figsize=(10, 10), save_path=None):
        plt.figure(figsize=figsize, dpi=60)
        sns.heatmap(data, xticklabels=False, yticklabels=False, cmap=camp,
                    center=0, annot=False, cbar=False, fmt='.2f')
        plt.savefig(save_path)
        plt.close()

    class Args:
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    args.norm_val = norm_val
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)

    data = args.source
    # print(f"Max grad before norm: {np.max(data)}")
    data = args.ALGRITHOM(data + 1)

    if args.norm_val is not None:
        data = data / args.norm_val
        # print(f"Using global norm value: {args.norm_val}")
    else:
        data = data / np.max(data)

    heatmap(data, save_path=args.heatmap_save)
    # print('heatmap saved at ', args.heatmap_save)


def visualize_erf(MODEL: nn.Module = None, num_images=100,
                  save_path=f"/tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
        grad = torch.autograd.grad(central_point, samples, create_graph=False)[0]

        # 【关键】防止深层梯度爆炸
        # grad = torch.clamp(grad, -1.0, 1.0)

        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    dataset = PairedImageDataset(opt)
    test_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = MODEL.cuda().eval()
    optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)
    meter = AverageMeter()
    optimizer.zero_grad()

    for idx, data_sample in enumerate(test_loader):
        if meter.count == num_images:
            return meter.avg
        samples = F.interpolate(data_sample['lq'], size=(160, 160))
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True
        optimizer.zero_grad()
        try:
            contribution_scores = get_input_grad(model, samples)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('OOM, skipping image')
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        torch.cuda.empty_cache()
        if np.isnan(np.sum(contribution_scores)):
            print('got NAN, next image')
            continue
        else:
            print(f'accumulat{idx}')
            meter.update(contribution_scores)

    return meter.avg


# ==========================
# 3. 模型包装器
# ==========================

class DeepLayerBranchVisualizer(nn.Module):
    """提取深层 MRFE 分支"""

    def __init__(self, original_model, layer_index=-1, block_index=-1, branch_id=0):
        super().__init__()
        self.original_model = original_model
        self.layer_index = layer_index
        self.block_index = block_index
        self.branch_id = branch_id
        self.model_mean = original_model.mean
        self.model_img_range = original_model.img_range
        self.conv_first = original_model.conv_first
        self.all_layers = original_model.layers

        target_layer = self.all_layers[self.layer_index]
        target_basic_layer = target_layer.residual_group
        self.target_block = target_basic_layer.blocks[self.block_index]

        self.norm1 = self.target_block.norm1
        self.ceb = self.target_block.CEB
        self.egem = self.target_block.EGEM
        self.isb = self.target_block.ISBlock
        self.skip_scale = self.target_block.skip_scale

    def forward(self, x):
        self.model_mean = self.model_mean.type_as(x)
        x = (x - self.model_mean) * self.model_img_range
        x = self.conv_first(x)

        for i, layer in enumerate(self.all_layers):
            if i < self.layer_index:
                x = layer(x)
            elif i == self.layer_index:
                blocks = layer.residual_group.blocks
                for j, block in enumerate(blocks):
                    if j < self.block_index:
                        x = block(x)
                    elif j == self.block_index:
                        x_norm = self.norm1(x.contiguous())
                        x_ceb = self.ceb(x_norm).contiguous()
                        split_indexes = self.isb.split_indexes
                        x_id, x_5, x_11, x_17 = torch.split(x_ceb, split_indexes, dim=1)

                        if self.branch_id == 0:
                            out = x_id
                        elif self.branch_id == 1:
                            out = self.isb.dwconv_hw(x_5)
                        elif self.branch_id == 2:
                            out = self.isb.dwconv_w(x_11)
                        elif self.branch_id == 3:
                            out = self.isb.dwconv_h(x_17)
                        else:
                            raise ValueError("Invalid branch_id")

                        return out
        return x


class OLKBVisualizer(nn.Module):
    """提取 OLKB 前后特征"""

    def __init__(self, original_model, layer_index=1, stage='after'):
        super().__init__()
        self.original_model = original_model
        self.layer_index = layer_index
        self.stage = stage
        self.model_mean = original_model.mean
        self.model_img_range = original_model.img_range
        self.conv_first = original_model.conv_first
        self.all_layers = original_model.layers

    def forward(self, x):
        self.model_mean = self.model_mean.type_as(x)
        x = (x - self.model_mean) * self.model_img_range
        x = self.conv_first(x)

        for i, layer in enumerate(self.all_layers):
            if i < self.layer_index:
                x = layer(x)
            elif i == self.layer_index:
                features_before_olkb = layer.residual_group(x)
                if self.stage == 'before':
                    return features_before_olkb
                elif self.stage == 'after':
                    features_after_olkb = layer.conv(features_before_olkb)
                    # 这里的逻辑对应 ResidualGroup 的输出：conv + residual
                    return features_after_olkb
        return x


# ==========================
# 4. 独立封装的主流程函数
# ==========================

def run_mrfe_visualization(model, layer_idx, block_index, save_base_path, num_images=50):
    """
    可视化 MRFE (Identity, 5x5, 11x11, 17x17) 的 ERF，并进行统一归一化。
    """
    print(f"\n{'=' * 10} Starting MRFE Visualization {'=' * 10}")
    print(f"Target: Layer {layer_idx}, Block {block_index}")

    branches = [
        (0, "Branch_Identity", "identity_erf.png"),
        (1, "Kernel_5x5", "5x5_erf.png"),
        (2, "Kernel_11x11", "11x11_erf.png"),
        (3, "Kernel_17x17", "17x17_erf.png")
    ]

    grad_maps = []
    configs = []

    # Phase 1: Compute
    for bid, kname, fname in branches:
        print(f"Calculating {kname}...")
        wrapper = DeepLayerBranchVisualizer(model, layer_index=layer_idx, block_index=block_index, branch_id=bid).cuda()
        save_dir = os.path.join(save_base_path, kname)
        os.makedirs(save_dir, exist_ok=True)
        temp_npy = f"./tmp2/{time.time()}_mrfe_{bid}_erf.npy"

        try:
            grad_map = visualize_erf(MODEL=wrapper, num_images=num_images, save_path=temp_npy)
            grad_maps.append(grad_map)
            configs.append((bid, kname, fname, save_dir))
            if os.path.exists(temp_npy): os.remove(temp_npy)
        except Exception as e:
            print(f"Error in {kname}: {e}")
            grad_maps.append(None)

    # Phase 2: Normalize & Plot
    print("\nNormalizing and saving heatmaps...")
    ALGRITHOM = lambda x: np.power(x - 1, 0.25)
    global_max_val = 0
    valid_indices = []

    for i, grad_map in enumerate(grad_maps):
        if grad_map is not None and not np.isnan(grad_map).any():
            val = np.max(ALGRITHOM(grad_map + 1))
            if val > global_max_val: global_max_val = val
            valid_indices.append(i)

    for i in valid_indices:
        bid, kname, fname, save_dir = configs[i]
        try:
            analyze_erf(source=grad_maps[i], dest=os.path.join(save_dir, fname), norm_val=global_max_val)
            print(f"Saved {kname}")
        except Exception as e:
            print(f"Error saving {kname}: {e}")


def run_olkb_visualization(model, layer_idx, save_base_path, num_images=50):
    """
    可视化 OLKB 前后的 ERF，并进行统一归一化。
    """
    print(f"\n{'=' * 10} Starting OLKB Visualization {'=' * 10}")
    print(f"Target: Layer {layer_idx}")

    stages = [
        ('before', "Before_OLKB", "erf_before_olkb.png"),
        ('after', "After_OLKB", "erf_after_olkb.png")
    ]

    grad_maps = []
    configs = []

    # Phase 1: Compute
    for stage, kname, fname in stages:
        print(f"Calculating {kname}...")
        wrapper = OLKBVisualizer(model, layer_index=layer_idx, stage=stage).cuda()
        save_dir = os.path.join(save_base_path, kname)
        os.makedirs(save_dir, exist_ok=True)
        temp_npy = f"./tmp2/{time.time()}_olkb_{stage}_erf.npy"

        try:
            grad_map = visualize_erf(MODEL=wrapper, num_images=num_images, save_path=temp_npy)
            grad_maps.append(grad_map)
            configs.append((stage, kname, fname, save_dir))
            if os.path.exists(temp_npy): os.remove(temp_npy)
        except Exception as e:
            print(f"Error in {kname}: {e}")
            grad_maps.append(None)

    # Phase 2: Normalize & Plot
    print("\nNormalizing and saving heatmaps...")
    ALGRITHOM = lambda x: np.power(x - 1, 0.25)
    global_max_val = 0
    valid_indices = []

    for i, grad_map in enumerate(grad_maps):
        if grad_map is not None and not np.isnan(grad_map).any():
            val = np.max(ALGRITHOM(grad_map + 1))
            if val > global_max_val: global_max_val = val
            valid_indices.append(i)

    for i in valid_indices:
        stage, kname, fname, save_dir = configs[i]
        try:
            analyze_erf(source=grad_maps[i], dest=os.path.join(save_dir, fname), norm_val=global_max_val)
            print(f"Saved {kname}")
        except Exception as e:
            print(f"Error saving {kname}: {e}")

#
# # ==========================
# # 5. 程序入口
# # ==========================
# if __name__ == '__main__':
#     import os
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     # 初始化模型
#     init_model = buildModel(4)
#     ckpt_path = r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x4_light_DF2K\models\net_g_505000.pth'
#     print(f"Loading checkpoint from {ckpt_path}")
#     init_model.load_state_dict(torch.load(ckpt_path)['params'])
#     init_model.eval()
#
#     # ==========================================
#     # 调用 1: MRFE 分支可视化
#     # ==========================================
#     run_mrfe_visualization(
#         model=init_model,
#         layer_idx=3,  # 目标层
#         block_index=4,  # 目标块
#         save_base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf_review_mrfe"),
#         num_images=50
#     )
#
#     # ==========================================
#     # 调用 2: OLKB 前后对比可视化
#     # ==========================================
#     run_olkb_visualization(
#         model=init_model,
#         layer_idx=3,  # 目标层
#         save_base_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf_review_olkb"),
#         num_images=50
#     )
#
#     print("\nAll tasks completed.")
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data as data
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from basicsr.data.data_util import paired_paths_from_folder
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils.matlab_functions import rgb2ycbcr
from timm.utils import AverageMeter
import pandas as pd
from typing import Dict, List, Tuple


# ========================== 核心ERF计算函数（固定4位小数） ==========================

def calculate_erf_metrics(grad_map: np.ndarray,
                          thresholds: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]) -> Dict[str, List[float]]:
    """
    基于像素计数计算ERF指标（严格≤100%）
    """
    h, w = grad_map.shape
    total_pixels = h * w

    # 计算每个像素到中心的欧氏距离
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    dist_map = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(np.float64)

    # 按距离排序
    flat_grad = grad_map.flatten().astype(np.float64)
    flat_dist = dist_map.flatten().astype(np.float64)
    sorted_indices = np.argsort(flat_dist)

    sorted_grad = flat_grad[sorted_indices]
    sorted_dist = flat_dist[sorted_indices]

    # 计算累积能量
    cumsum_grad = np.cumsum(sorted_grad)
    total_grad = cumsum_grad[-1]

    if total_grad == 0:
        return {
            'thresholds': thresholds,
            'side_lengths': [0.0] * len(thresholds),
            'area_ratios': [0.0] * len(thresholds)
        }

    # 计算各阈值指标
    side_lengths = []
    area_ratios = []

    for th in thresholds:
        th = max(0.0, min(1.0, th))
        target_energy = total_grad * th
        idx = np.searchsorted(cumsum_grad, target_energy)

        # 像素计数法（精确）
        covered_pixels = idx + 1
        area_ratio = (covered_pixels / total_pixels) * 100.0

        radius = sorted_dist[idx]
        side_length = 2 * radius

        side_lengths.append(float(side_length))
        area_ratios.append(float(min(100.0, area_ratio)))

    return {
        'thresholds': thresholds,
        'side_lengths': side_lengths,
        'area_ratios': area_ratios
    }
def plot_erf_cdf(metrics_list: List[Tuple[str, Dict]],
                 thresholds: List[float],
                 save_path: str = None):
    """
    绘制ERF面积的累积分布函数（CDF）。
    metrics_list: List of tuples [(name, metrics_dict)]
    thresholds: 阈值列表，例如 [0.1, 0.2, ..., 0.95]
    save_path: 可选，保存路径
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6), dpi=100)

    for name, metrics in metrics_list:
        if metrics is None:
            continue
        # area_ratios就是累积百分比
        cdf_vals = metrics['area_ratios']
        plt.plot([t*100 for t in thresholds], cdf_vals, marker='o', label=name)

    plt.xlabel("Gradient Energy Threshold (%)", fontsize=14)
    plt.ylabel("ERF Area Covered (%)", fontsize=14)
    plt.title("ERF CDF Curve", fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"✓ ERF CDF saved at: {save_path}")
    plt.show()


# ========================== 固定4位小数格式化 ==========================
def format_pct(value: float) -> str:
    """统一4位小数格式化"""
    return f"{value:.4f}%"


# ========================== 表格打印函数（4位小数） ==========================
def print_erf_metrics_table(metrics_list: List[Tuple[str, Dict]],
                            thresholds: List[float]):
    """
    打印ERF指标表格（统一4位小数）
    """
    print("\n" + "=" * 90)
    header = f"{'Component':<25}"
    for th in thresholds:
        th_pct = int(round(th * 100))
        header += f" | Th_{th_pct}%"
    print(header)
    print("-" * 90)

    for name, metrics in metrics_list:
        if not metrics:
            continue
        row = f"{name:<25}"
        for area in metrics['area_ratios']:
            row += f" | {format_pct(area):>10}"
        print(row)

    print("=" * 90)


# ========================== OLKB变化分析（4位小数） ==========================
def print_olkb_change_analysis(before_metrics: Dict, after_metrics: Dict, thresholds: List[float]):
    """
    OLKB前后变化分析（统一4位小数）
    """
    print("\n→ OLKB Impact Analysis (Fixed 4-decimal precision):")
    header = f"{'Th':<6} | {'Before':<12} | {'After':<12} | {'Δ Abs':<12} | {'Δ Rel(%)':<12} | {'Interpretation':<20}"
    print(header)
    print("-" * 90)

    for i, th in enumerate(thresholds):
        th_pct = int(round(th * 100))
        before_area = before_metrics['area_ratios'][i]
        after_area = after_metrics['area_ratios'][i]

        # 绝对变化（百分点）
        abs_change = after_area - before_area

        # 相对变化（百分比）
        rel_change = ((after_area - before_area) / before_area * 100.0) if before_area > 1e-8 else 0.0

        # 简单解释逻辑
        if abs(abs_change) < 0.01:
            interp = "Negligible"
        elif abs_change < -0.1:
            interp = "Strong contraction"
        elif abs_change < 0:
            interp = "Mild contraction"
        elif abs_change < 0.1:
            interp = "Stable"
        else:
            interp = "Expansion"

        print(f"{th_pct}%{'':<3} | {format_pct(before_area):<12} | {format_pct(after_area):<12} | "
              f"{abs_change:+.4f}pp{'':<4} | {rel_change:+.4f}%{'':<5} | {interp:<20}")

    print("-" * 90)
    print("💡 Notes:")
    print("   • Δ Abs: Absolute change in percentage points (pp)")
    print("   • Δ Rel: Relative change percentage")
    print("   • Changes < 0.01pp considered negligible (numerical noise)")


# ========================== CSV导出（4位小数） ==========================
def export_erf_metrics_csv(metrics_list: List[Tuple[str, Dict]],
                           save_path: str,
                           thresholds: List[float],
                           input_size: Tuple[int, int] = (160, 160)):
    """
    导出ERF指标到CSV（统一4位小数）
    """
    rows = []
    for name, metrics in metrics_list:
        row = {'Component': name, 'Input_Size': f"{input_size[0]}x{input_size[1]}"}
        for i, th in enumerate(thresholds):
            th_pct = int(round(th * 100))
            row[f'Th_{th_pct}%_SideLength_px'] = f"{metrics['side_lengths'][i]:.4f}"
            row[f'Th_{th_pct}%_AreaRatio_pct'] = f"{metrics['area_ratios'][i]:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✓ ERF metrics exported to: {save_path} (4-decimal precision)")
    return df


# ========================== 可视化函数 ==========================
def visualize_erf_with_metrics(MODEL: nn.Module,
                               num_images: int,
                               component_name: str,
                               thresholds: List[float]) -> Tuple[np.ndarray, Dict]:
    """
    计算ERF并返回量化指标
    """

    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
        grad = torch.autograd.grad(central_point, samples, create_graph=False)[0]
        grad = torch.nn.functional.relu(grad)
        return grad.sum((0, 1)).cpu().numpy()

    dataset = PairedImageDataset(opt)
    test_loader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = MODEL.cuda().eval()
    meter = AverageMeter()

    for idx, data_sample in enumerate(test_loader):
        if meter.count >= num_images:
            break

        samples = F.interpolate(data_sample['lq'], size=(160, 160))
        samples = samples.cuda(non_blocking=True)
        samples.requires_grad = True

        try:
            contribution_scores = get_input_grad(model, samples)
            torch.cuda.empty_cache()

            if not np.isnan(np.sum(contribution_scores)):
                meter.update(contribution_scores)
                print(f"  [{component_name}] Accumulated {meter.count}/{num_images} images")
            else:
                print(f"  [{component_name}] Skipping NaN gradient")

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"  [{component_name}] OOM, skipping image")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

    grad_map = meter.avg
    metrics = calculate_erf_metrics(grad_map, thresholds=thresholds)

    # 高精度打印（4位小数）
    print(f"  [{component_name}] ERF Metrics (160×160 input):")
    for i, th in enumerate(metrics['thresholds']):
        th_pct = int(round(th * 100))
        area = metrics['area_ratios'][i]
        side = metrics['side_lengths'][i]
        print(f"    → Th_{th_pct}%: {area:.4f}% area | {side:.2f}px side")

    return grad_map, metrics


# ========================== 主流程函数 ==========================
def run_mrfe_visualization_with_metrics(model,
                                        layer_idx,
                                        block_index,
                                        save_base_path,
                                        num_images=50,
                                        thresholds: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]):
    """
    MRFE分支ERF分析（4位小数）
    """
    print(f"\n{'=' * 90}")
    print(f"MRFE ERF Analysis: Layer {layer_idx}, Block {block_index}")
    print(f"Thresholds: {[f'{t * 100:.0f}%' for t in thresholds]} | Precision: 4 decimal places")
    print(f"{'=' * 90}")

    branches = [
        (0, "Identity", "identity_erf.png"),
        (1, "5×5_Kernel", "5x5_erf.png"),
        (2, "11×11_Kernel", "11x11_erf.png"),
        (3, "17×17_Kernel", "17x17_erf.png")
    ]

    grad_maps = []
    metrics_list = []
    configs = []

    # Phase 1: 计算ERF
    for bid, kname, fname in branches:
        print(f"\n→ Processing {kname} (Branch ID={bid})...")
        wrapper = DeepLayerBranchVisualizer(model, layer_index=layer_idx, block_index=block_index, branch_id=bid).cuda()
        save_dir = os.path.join(save_base_path, kname.replace('×', 'x'))
        os.makedirs(save_dir, exist_ok=True)

        try:
            grad_map, metrics = visualize_erf_with_metrics(wrapper, num_images, kname, thresholds)
            grad_maps.append(grad_map)
            metrics_list.append((kname, metrics))
            configs.append((bid, kname, fname, save_dir))
        except Exception as e:
            print(f"  ✗ Error in {kname}: {e}")
            grad_maps.append(None)
            metrics_list.append((kname, None))

    # Phase 2: 可视化
    print("\n→ Normalizing and saving heatmaps...")
    ALGRITHOM = lambda x: np.power(x - 1, 0.25)
    global_max_val = 0
    valid_indices = []

    for i, grad_map in enumerate(grad_maps):
        if grad_map is not None and not np.isnan(grad_map).any():
            val = np.max(ALGRITHOM(grad_map + 1))
            global_max_val = max(global_max_val, val)
            valid_indices.append(i)

    for i in valid_indices:
        bid, kname, fname, save_dir = configs[i]
        try:
            analyze_erf(source=grad_maps[i], dest=os.path.join(save_dir, fname), norm_val=global_max_val)
            print(f"  ✓ Saved heatmap: {kname}")
        except Exception as e:
            print(f"  ✗ Error saving {kname}: {e}")

    # Phase 3: 导出指标
    print("\n→ Exporting ERF metrics...")
    csv_path = os.path.join(save_base_path, f"MRFE_L{layer_idx}_B{block_index}_metrics.csv")
    valid_metrics = [(name, m) for name, m in metrics_list if m is not None]
    df = export_erf_metrics_csv(valid_metrics, csv_path, thresholds, input_size=(160, 160))
    print_erf_metrics_table(valid_metrics, thresholds)

    # Phase 4: 绘制CDF曲线
    cdf_save_path = os.path.join(save_base_path, f"MRFE_L{layer_idx}_B{block_index}_CDF.png")
    plot_erf_cdf(valid_metrics, thresholds, save_path=cdf_save_path)
    return df


def run_olkb_visualization_with_metrics(model,
                                        layer_idx,
                                        save_base_path,
                                        num_images=50,
                                        thresholds: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]):
    """
    OLKB前后ERF对比分析（4位小数）
    """
    print(f"\n{'=' * 90}")
    print(f"OLKB ERF Analysis: Layer {layer_idx}")
    print(f"Thresholds: {[f'{t * 100:.0f}%' for t in thresholds]} | Precision: 4 decimal places")
    print(f"{'=' * 90}")

    stages = [
        ('before', "Before_OLKB", "erf_before_olkb.png"),
        ('after', "After_OLKB", "erf_after_olkb.png")
    ]

    grad_maps = []
    metrics_list = []
    configs = []

    # Phase 1: 计算ERF
    for stage, kname, fname in stages:
        print(f"\n→ Processing {kname} (stage={stage})...")
        wrapper = OLKBVisualizer(model, layer_index=layer_idx, stage=stage).cuda()
        save_dir = os.path.join(save_base_path, kname)
        os.makedirs(save_dir, exist_ok=True)

        try:
            grad_map, metrics = visualize_erf_with_metrics(wrapper, num_images, kname, thresholds)
            grad_maps.append(grad_map)
            metrics_list.append((kname, metrics))
            configs.append((stage, kname, fname, save_dir))
        except Exception as e:
            print(f"  ✗ Error in {kname}: {e}")
            grad_maps.append(None)
            metrics_list.append((kname, None))

    # Phase 2: 可视化
    print("\n→ Normalizing and saving heatmaps...")
    ALGRITHOM = lambda x: np.power(x - 1, 0.25)
    global_max_val = 0
    valid_indices = []

    for i, grad_map in enumerate(grad_maps):
        if grad_map is not None and not np.isnan(grad_map).any():
            val = np.max(ALGRITHOM(grad_map + 1))
            global_max_val = max(global_max_val, val)
            valid_indices.append(i)

    for i in valid_indices:
        stage, kname, fname, save_dir = configs[i]
        try:
            analyze_erf(source=grad_maps[i], dest=os.path.join(save_dir, fname), norm_val=global_max_val)
            print(f"  ✓ Saved heatmap: {kname}")
        except Exception as e:
            print(f"  ✗ Error saving {kname}: {e}")

    # Phase 3: 导出指标 + 变化分析
    print("\n→ Exporting ERF metrics...")
    csv_path = os.path.join(save_base_path, f"OLKB_L{layer_idx}_metrics.csv")
    valid_metrics = [(name, m) for name, m in metrics_list if m is not None]
    df = export_erf_metrics_csv(valid_metrics, csv_path, thresholds, input_size=(160, 160))
    print_erf_metrics_table(valid_metrics, thresholds)

    # 变化分析
    if len(valid_metrics) == 2:
        print_olkb_change_analysis(valid_metrics[0][1], valid_metrics[1][1], thresholds)

    return df


# ========================== 程序入口 ==========================
if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 初始化模型
    init_model = buildModel(2)
    # ckpt_path = r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x4_light_DF2K\models\net_g_505000.pth'
    # ckpt_path = r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x2_light_Div2K\models\net_g_500000.pth'
    # ckpt_path = r'F:\code\CodeProject\HLKNet\analysis\ckpt\echoSR_review\COFB15-7_505000.pth'

    ckpt_path=r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x2_light_Div2K_5-15-25\models\net_g_520000.pth'
    print(f"Loading checkpoint from {ckpt_path}")
    init_model.load_state_dict(torch.load(ckpt_path)['params'])
    init_model.eval()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 统一阈值配置（4位小数精度）
    THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95]

    # MRFE分析
    print("\n" + "#" * 90)
    print("# MRFE Branch ERF Quantification (4-decimal precision)")
    print("#" * 90)
    mrfe_df = run_mrfe_visualization_with_metrics(
        model=init_model,
        layer_idx=3,
        block_index=4,
        save_base_path=os.path.join(base_dir, "show/erf_review/5-15-25_mrfe"),
        num_images=50,
        thresholds=THRESHOLDS
    )

    # OLKB分析
    print("\n" + "#" * 90)
    print("# OLKB Module ERF Quantification (4-decimal precision)")
    print("#" * 90)
    olkb_df = run_olkb_visualization_with_metrics(
        model=init_model,
        layer_idx=3,
        save_base_path=os.path.join(base_dir, "show/erf_review/5-15-25_olkb"),
        num_images=50,
        thresholds=THRESHOLDS
    )

    # 完成提示
    print("\n" + "#" * 90)
    print("# ERF Analysis Completed Successfully")
    print("#" * 90)
    print(f"✓ All metrics reported with 4-decimal precision (e.g., 0.0612%)")
    print(f"✓ CSV files contain 4-decimal values for direct paper table generation")
    print(f"✓ Terminal tables and change analysis use consistent 4-decimal formatting")
    print("\n✅ Ready for paper writing and reviewer response.")