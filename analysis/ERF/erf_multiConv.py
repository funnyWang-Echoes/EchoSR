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
root_path = r'F:\code\CodeProject\HLKNet\options\HLKNet_test\test_HLKNet_light_SRx4.yml'
opt, _ = parse_options(root_path, is_train=False)
opt=opt['datasets']['test_1'] # we use the 4-th SR testsets(i.e. Urban100) to visualize ERF.


class PairedImageDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        # 【关键修复】不要直接引用 self.io_backend_opt，而是复制一份或单独存储 type
        self.io_backend_opt = opt['io_backend']
        self.io_backend_type = self.io_backend_opt.get('type', 'disk')  # 安全获取 type

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
            # 【关键修复】使用预先保存的 type，并创建一个临时字典副本以避免修改全局 opt
            # 这样即使 pop 操作执行，也不会影响原始的 opt 字典
            temp_io_args = self.io_backend_opt.copy()
            backend_type = temp_io_args.pop('type', self.io_backend_type)
            self.file_client = FileClient(backend_type, **temp_io_args)

        scale = self.opt['scale']

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;

        if self.task == 'CAR':
            # image range: [0, 255], int., H W 1
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, flag='grayscale', float32=False)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, flag='grayscale', float32=False)
            img_gt = np.expand_dims(img_gt, axis=2).astype(np.float32) / 255.
            img_lq = np.expand_dims(img_lq, axis=2).astype(np.float32) / 255.

        elif self.task == 'denoising_gray':  # Matlab + OpenCV version
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
            # image range: [0, 1], float32., H W 3
            gt_path = self.paths[index]['gt_path']
            img_bytes = self.file_client.get(gt_path, 'gt')
            img_gt = imfrombytes(img_bytes, float32=True)
            lq_path = self.paths[index]['lq_path']
            img_bytes = self.file_client.get(lq_path, 'lq')
            img_lq = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            # random crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)
            # flip, rotation
            img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        # crop the unmatched GT images during validation or testing, especially for SR benchmark datasets
        # TODO: It is better to update the datasets, rather than force to crop
        if self.opt['phase'] != 'train':
            img_gt = img_gt[0:img_lq.shape[0] * scale, 0:img_lq.shape[1] * scale, :]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)

        return {'lq': img_lq, 'gt': img_gt, 'lq_path': lq_path, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)



if True:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.family"] = "Times New Roman"
    import seaborn as sns

    #   Set figure parameters
    large = 24;
    med = 24;
    small = 24
    params = {'axes.titlesize': large,
              'legend.fontsize': med,
              'figure.figsize': (16, 10),
              'axes.labelsize': med,
              'xtick.labelsize': med,
              'ytick.labelsize': med,
              'figure.titlesize': large}
    plt.rcParams.update(params)
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_style("white")
    # plt.rc('font', **{'family': 'Times New Roman'})
    plt.rcParams['axes.unicode_minus'] = False






# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def analyze_erf(source, dest="heatmap.png", ALGRITHOM=lambda x: np.power(x - 1, 0.25)):
    def heatmap(data, camp='BrBG', figsize=(10, 10), ax=None, save_path=None):
        camp = 'RdYlGn'
        plt.figure(figsize=figsize, dpi=60)
        ax = sns.heatmap(data,
                         xticklabels=False,
                         yticklabels=False, cmap=camp,
                         center=0, annot=False, ax=ax, cbar=False, annot_kws={"size": 24}, fmt='.2f')
        plt.savefig(save_path)

    def analyze_erf(args):
        data = args.source
        print(np.max(data))
        print(np.min(data))
        data = args.ALGRITHOM(data + 1)  # the scores differ in magnitude. take the logarithm for better readability
        data = data / np.max(data)  # rescale to [0,1] for the comparability among models
        heatmap(data, save_path=args.heatmap_save)
        print('heatmap saved at ', args.heatmap_save)

    class Args():
        ...

    args = Args()
    args.source = source
    args.heatmap_save = dest
    args.ALGRITHOM = ALGRITHOM
    os.makedirs(os.path.dirname(args.heatmap_save), exist_ok=True)
    analyze_erf(args)


# copied from https://github.com/DingXiaoH/RepLKNet-pytorch
def visualize_erf(MODEL: nn.Module = None, num_images=100, data_path="/mnt/data/WorkSpace/wbh/myMambaIR/datasets/SR/Urban100/LR_bicubic",
                  save_path=f"/tmp/{time.time()}/erf.npy"):
    def get_input_grad(model, samples):
        outputs = model(samples)
        out_size = outputs.size()
        central_point = outputs[:, :, out_size[2] // 2, out_size[3] // 2].sum()
        grad = torch.autograd.grad(central_point, samples)
        grad = grad[0]
        grad = torch.nn.functional.relu(grad)
        aggregated = grad.sum((0, 1))
        grad_map = aggregated.cpu().numpy()
        return grad_map

    def main(args, MODEL: nn.Module = None):
        print("reading from datapath", args.data_path)
        root = args.data_path
        dataset = PairedImageDataset(opt)

        test_loader = data.DataLoader(dataset,batch_size=1,shuffle=False)

        model = MODEL
        model.cuda().eval()

        optimizer = optim.SGD(model.parameters(), lr=0, weight_decay=0)

        meter = AverageMeter()
        optimizer.zero_grad()

        for idx,data_sample in enumerate(test_loader):
            if meter.count == args.num_images:
                return meter.avg
            # we set the imhg size to 120X120 due to the GPU memory constrain
            samples = F.interpolate(data_sample['lq'],size=(160,160))
            samples = samples.cuda(non_blocking=True)
            samples.requires_grad = True
            optimizer.zero_grad()
            contribution_scores = get_input_grad(model, samples)
            torch.cuda.empty_cache()
            if np.isnan(np.sum(contribution_scores)):
                print('got NAN, next image')
                continue
            else:
                print(f'accumulat{idx}')
                meter.update(contribution_scores)

        return meter.avg


    class Args():
        ...

    args = Args()
    args.num_images = num_images
    args.data_path = data_path
    args.save_path = save_path
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    return main(args, MODEL)


class BranchVisualizer(nn.Module):
    """
    包装器模型，用于提取 EchoSR 中 InceptionStyleDWConv2d 特定分支的输出。
    """

    def __init__(self, original_model, branch_id):
        super().__init__()
        self.original_model = original_model
        self.branch_id = branch_id  # 0: 5x5, 1: 11x11, 2: 17x17

        # 获取第一层 ResidualGroup 中的第一个 HybridInceptionBlock
        # 注意：这里假设你的模型结构至少有一层
        first_layer = original_model.layers[0]
        first_block = first_layer.residual_group.blocks[0]

        # 缓存需要用到的子模块
        self.conv_first = original_model.conv_first
        self.norm1 = first_block.norm1
        self.ceb = first_block.CEB
        self.egem = first_block.EGEM
        self.isb = first_block.ISBlock
        self.skip_scale = first_block.skip_scale

    def forward(self, x):
        # 1. 浅层特征提取
        x = self.conv_first(x)

        # 2. 进入第一个 Block
        x = self.norm1(x)

        # Local Aggregation (LA)
        x_ceb = self.ceb(x)

        # Global Context Estimation (GCE)
        # 注意：为了纯粹看分支的感受野，我们这里只取 ISBlock 的分支输出
        # 如果你想看 GCE + 某个分支的效果，可以在这里加上 self.egem(x_ceb)

        # 3. InceptionStyleDWConv2d (ISB) 前向逻辑手动实现
        # ISB 的 forward: x_id, x_5, x_11, x_17 = torch.split(x, self.split_indexes, dim=1)
        split_indexes = self.isb.split_indexes
        x_id, x_5, x_11, x_17 = torch.split(x_ceb, split_indexes, dim=1)

        if self.branch_id == 0:
            # 5x5 branch
            out = self.isb.dwconv_hw(x_5)
            print("Visualizing 5x5 branch ERF...")
        elif self.branch_id == 1:
            # 11x11 branch
            out = self.isb.dwconv_w(x_11)
            print("Visualizing 11x11 branch ERF...")
        elif self.branch_id == 2:
            # 17x17 branch
            out = self.isb.dwconv_h(x_17)
            print("Visualizing 17x17 branch ERF...")
        else:
            raise ValueError("Invalid branch_id. Must be 0, 1, or 2.")

        return out


class DeepLayerBranchVisualizer(nn.Module):
    """
    包装器模型，完整复现 HLKNet 的输入预处理，并提取深层分支的 ERF。
    """

    def __init__(self, original_model, layer_index=-1, block_index=-1, branch_id=0):
        super().__init__()
        self.original_model = original_model
        self.layer_index = layer_index
        self.block_index = block_index
        self.branch_id = branch_id  # 0: 5x5, 1: 11x11, 2: 17x17

        # --- 1. 获取预处理参数 ---
        # HLKNet 的 mean 和 img_range 是预处理的关键
        self.model_mean = original_model.mean
        self.model_img_range = original_model.img_range
        self.conv_first = original_model.conv_first

        # --- 2. 定位目标 Block ---
        self.all_layers = original_model.layers  # list of ResidualGroup

        # 获取目标层
        target_layer = self.all_layers[self.layer_index]
        # 获取目标层内的 BasicLayer
        target_basic_layer = target_layer.residual_group
        # 获取目标 Block
        self.target_block = target_basic_layer.blocks[self.block_index]

        # 缓存目标 Block 内部模块
        self.norm1 = self.target_block.norm1
        self.ceb = self.target_block.CEB
        self.egem = self.target_block.EGEM
        self.isb = self.target_block.ISBlock
        self.skip_scale = self.target_block.skip_scale

    def forward(self, x):
        # --- 步骤 A: 复现 HLKNet 的输入预处理 (关键修复) ---
        self.model_mean = self.model_mean.type_as(x)
        x = (x - self.model_mean) * self.model_img_range

        # --- 步骤 B: 浅层特征提取 ---
        x = self.conv_first(x)

        # --- 步骤 C: 逐层传播直到目标层 ---
        for i, layer in enumerate(self.all_layers):
            # 如果在目标层之前，完整执行该层的 ResidualGroup 逻辑
            if i < self.layer_index:
                # ResidualGroup.forward: return self.conv(self.residual_group(x)) + x
                x = layer(x)

            # 如果到达目标层
            elif i == self.layer_index:
                # 获取当前层的所有 blocks
                # 注意：ResidualGroup 的 residual_group 属性是 BasicLayer
                blocks = layer.residual_group.blocks

                for j, block in enumerate(blocks):
                    # 如果在目标 Block 之前，完整执行 Block 逻辑
                    if j < self.block_index:
                        x = block(x)

                    # 如果正好到达目标 Block
                    elif j == self.block_index:
                        # --- 步骤 D: 手动执行 Block 逻辑并进行拦截 ---

                        # 1. Norm + LA (Local Aggregation)
                        x_norm = self.norm1(x.contiguous())
                        x_ceb = self.ceb(x_norm).contiguous()

                        # 2. ISB 分支逻辑
                        # 在这里我们只提取特定分支的输出
                        split_indexes = self.isb.split_indexes
                        x_id, x_5, x_11, x_17 = torch.split(x_ceb, split_indexes, dim=1)

                        if self.branch_id == 0:
                            out = self.isb.dwconv_hw(x_5)
                            print(f"Visualizing 5x5 branch ERF at Deep Layer {i} Block {j}...")
                        elif self.branch_id == 1:
                            out = self.isb.dwconv_w(x_11)
                            print(f"Visualizing 11x11 branch ERF at Deep Layer {i} Block {j}...")
                        elif self.branch_id == 2:
                            out = self.isb.dwconv_h(x_17)
                            print(f"Visualizing 17x17 branch ERF at Deep Layer {i} Block {j}...")
                        else:
                            raise ValueError("Invalid branch_id.")

                        # 直接返回该分支的输出，让梯度从这里往回传
                        return out

                    else:
                        # 理论上不会走到这里，因为上面 return 了
                        x = block(x)

            else:
                # 理论上不会走到这里
                x = layer(x)

        return x


if __name__ == '__main__':
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 配置路径
    base_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf_last_layer_branches")

    # 初始化模型
    init_model = buildModel(4)

    # 加载预训练权重
    ckpt_path = r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x4_light_DF2K\models\net_g_505000.pth'
    print(f"Loading checkpoint from {ckpt_path}")
    init_model.load_state_dict(torch.load(ckpt_path)['params'])
    init_model.eval()

    # 模型结构参数 (根据你的 buildModel 定义)
    # depths=(5, 5, 5, 5,) -> 4层，每层5个block
    # 最后一层索引为 3 (或 -1)
    # 最后一个block索引为 4 (或 -1)
    target_layer_idx = 1
    target_block_idx = 0

    branches_to_vis = [
        (0, "Kernel_5x5_Deep", "5x5_erf.png"),
        (1, "Kernel_11x11_Deep", "11x11_erf.png"),
        (2, "Kernel_17x17_Deep", "17x17_erf.png")
    ]

    for bid, kname, fname in branches_to_vis:
        print(f"\n=== Starting Visualization for {kname} (Last Layer, Last Block) ===")

        # 1. 创建深层可视化包装器
        branch_model = DeepLayerBranchVisualizer(
            init_model,
            layer_index=target_layer_idx,
            block_index=target_block_idx,
            branch_id=bid
        ).cuda()

        # 2. 运行可视化流程
        save_dir = os.path.join(base_save_path, kname)
        os.makedirs(save_dir, exist_ok=True)

        temp_npy = f"./tmp2/{time.time()}_deep_branch{bid}_erf.npy"

        try:
            # 注意：由于计算图更深（经过了所有层），显存占用可能会稍微增加，
            # 如果 OOM，可以减小 visualize_erf 中的 size=(160,160) 或者减少 num_images
            grad_map = visualize_erf(MODEL=branch_model, num_images=50, save_path=temp_npy)

            analyze_erf(source=grad_map, dest=os.path.join(save_dir, fname))
            print(f"Saved {kname} visualization to {save_dir}")

            if os.path.exists(temp_npy):
                os.remove(temp_npy)

        except RuntimeError as e:
            if 'out of memory' in str(e):
                print(f"OOM Error for {kname}: GPU memory exhausted. Try reducing image size or num_images.")
                if os.path.exists(temp_npy): os.remove(temp_npy)
            else:
                raise e
        except Exception as e:
            print(f"Error visualizing {kname}: {e}")

    print("\nAll visualizations completed.")
# if __name__ == '__main__':
#     import os
#     import shutil
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
#     # 配置路径
#     # 建议为每个核创建单独的文件夹保存结果
#     base_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "show/erf_branches")
#
#     # 初始化模型
#     # 注意：请确保你的 buildModel 函数定义在上方或已导入
#     init_model = buildModel(4)
#
#     # 加载预训练权重
#     ckpt_path = r'F:\code\CodeProject\HLKNet\experiments\HLKNet_SR_x4_light_DF2K\models\net_g_505000.pth'
#     print(f"Loading checkpoint from {ckpt_path}")
#     init_model.load_state_dict(torch.load(ckpt_path)['params'])
#     init_model.eval()
#
#     # 我们要可视化的分支配置
#     # (Branch ID, Kernel Size Name, Output Filename)
#     branches_to_vis = [
#         (0, "Kernel_5x5", "5x5_erf.png"),
#         (1, "Kernel_11x11", "11x11_erf.png"),
#         (2, "Kernel_17x17", "17x17_erf.png")
#     ]
#
#     for bid, kname, fname in branches_to_vis:
#         print(f"\n=== Starting Visualization for {kname} ===")
#
#         # 1. 创建分支可视化包装器
#         branch_model = BranchVisualizer(init_model, branch_id=bid).cuda()
#
#         # 2. 运行可视化流程
#         # 使用你提供的 visualize_erf 函数，但传入我们的 branch_model
#         # 注意：visualize_erf 内部会自动处理数据和梯度计算
#         save_dir = os.path.join(base_save_path, kname)
#         os.makedirs(save_dir, exist_ok=True)
#
#         # 传入临时保存路径
#         temp_npy = f"./tmp2/{time.time()}_branch{bid}_erf.npy"
#
#         try:
#             # 调用你原有的函数
#             grad_map = visualize_erf(MODEL=branch_model, num_images=100, save_path=temp_npy)
#
#             # 3. 生成热力图
#             # 稍微调整一下 ALGRITHOM 参数以适应单层特征（单层特征可能比较稀疏，线性显示可能更好看，或者沿用默认的次幂）
#             # 这里沿用你的默认设置：lambda x: np.power(x - 1, 0.25)
#             analyze_erf(source=grad_map, dest=os.path.join(save_dir, fname))
#             print(f"Saved {kname} visualization to {save_dir}")
#
#             # 清理临时文件
#             if os.path.exists(temp_npy):
#                 os.remove(temp_npy)
#
#         except Exception as e:
#             print(f"Error visualizing {kname}: {e}")
#
#     print("\nAll visualizations completed.")



