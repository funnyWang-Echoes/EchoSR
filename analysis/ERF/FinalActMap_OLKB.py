import os
import cv2
import glob
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from model_zoo.myModel import buildModel


# 启用 LaTeX 排版
plt.rcParams['text.usetex'] = True
# 设置全局字体为 Times New Roman 并增大字体大小
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Computer Modern Roman'] # 设置字体为 Times New Roman
rcParams['font.size'] = 40 # 增大字体大小
# # 设置全局字体为 Times New Roman 并增大字体大小
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
# rcParams['font.size'] = 40  # 增大字体大小

# 多分支大核卷积
def save_activation_mean_maps(activations_list, imgnames, save_dir, prefix, original_images):
    """Save activation maps from each branch for multiple images in a single figure."""

    num_images = len(imgnames)
    branch_names = ['Image', 'Identity', 'Kernel size = 5', 'Kernel size = 11', 'Kernel size = 17']

    # 创建一个大图，每个图片一行，每行5个子图（原图+4个激活图），最右侧一列用于颜色条
    fig, axs = plt.subplots(num_images, 6, figsize=(30, 5 * num_images),
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.05]})

    # 如果只有一张图片，将axs转换为二维数组以保持一致的索引方式
    if num_images == 1:
        axs = np.array([axs])

    # 收集所有激活图以进行全局归一化
    all_activations = []

    # 首先收集所有激活值以进行全局归一化
    for i, (x_id, x_hw, x_w, x_h, x_egem) in enumerate(activations_list):
        for branch in [x_id, x_hw, x_w, x_h]:
            if branch is not None:
                avg_activation = torch.mean(branch, dim=1, keepdim=True)
                img_np = avg_activation.squeeze().float().cpu().numpy()
                img_np = np.maximum(img_np, 1e-6)  # 处理负值
                img_np = np.log1p(img_np)  # 应用对数变换
                all_activations.append(img_np)

    # 全局归一化
    all_activations = np.array(all_activations)
    vmin, vmax = np.min(all_activations), np.max(all_activations)

    # 处理每张图片
    for i, ((x_id, x_hw, x_w, x_h, x_egem), imgname, original_image) in enumerate(
            zip(activations_list, imgnames, original_images)):
        # 显示原始图像
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axs[i, 0].imshow(original_image)
        axs[i, 0].axis('off')

        # 处理激活图
        branches = [x_id, x_hw, x_w, x_h]

        for j, branch in enumerate(branches):
            if branch is not None:
                avg_activation = torch.mean(branch, dim=1, keepdim=True)
                img_np = avg_activation.squeeze().float().cpu().numpy()
                img_np = np.maximum(img_np, 1e-6)  # 处理负值
                img_np = np.log1p(img_np)  # 应用对数变换

                # 归一化到 [0, 1] 范围（使用全局最大最小值）
                img_np = (img_np - vmin) / (vmax - vmin + 1e-8)

                # 增强对比度
                img_np = np.power(img_np, 0.5)  # 调整此值以控制对比度

                im = axs[i, j + 1].imshow(img_np, cmap='viridis', vmin=0, vmax=1)
                axs[i, j + 1].axis('off')
            else:
                axs[i, j + 1].axis('off')

        # 关闭每行最后一个子图（颜色条位置）的显示
        axs[i, -1].axis('off')

    # 创建一个单独的颜色条，跨越所有行
    cbar_ax = fig.add_axes([0.90, 0.15, 0.01, 0.69])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Activation intensity')

    # 减少图片间的间距
    plt.subplots_adjust(wspace=0.01, hspace=-0.15)

    # 添加标签到图像下方
    # 计算每列的中心位置
    col_positions = []
    for j in range(5):
        bbox = axs[-1, j].get_position()
        col_center = (bbox.x0 + bbox.x1) / 2
        col_positions.append(col_center)

    # 在图像底部添加标签
    for j in range(5):
        fig.text(col_positions[j], 0.12, branch_names[j], ha='center', va='center', fontsize=32)

    # 保存图片
    pdf_path = os.path.join(save_dir, f'{prefix}_multiple_activations.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()

# 重叠大核卷积
def save_before_after_activations(before_activations, after_activations, imgnames, save_dir, original_images):
    """Save before and after conv activation maps."""

    num_images = len(imgnames)

    # 创建一个大图，调整figsize和子图间距
    fig, axs = plt.subplots(num_images, 4, figsize=(18, 5 * num_images),  # 减小总宽度
                            gridspec_kw={
                                'width_ratios': [1, 1, 1, 0.05],  # 三列图片+颜色条
                                'wspace': 0.05,  # 初始子图水平间距
                                'hspace': 0.02  # 初始子图垂直间距
                            })

    # 如果只有一张图片，将axs转换为二维数组以保持一致的索引方式
    if num_images == 1:
        axs = np.array([axs])

    # 收集所有激活图以进行全局归一化
    all_activations = []

    # 收集前后激活值
    for before, after in zip(before_activations, after_activations):
        if before is not None:
            avg_activation = torch.mean(before, dim=1, keepdim=True)
            img_np = avg_activation.squeeze().float().cpu().numpy()
            img_np = np.maximum(img_np, 1e-6)
            img_np = np.log1p(img_np)
            all_activations.append(img_np)

        if after is not None:
            avg_activation = torch.mean(after, dim=1, keepdim=True)
            img_np = avg_activation.squeeze().float().cpu().numpy()
            img_np = np.maximum(img_np, 1e-6)
            img_np = np.log1p(img_np)
            all_activations.append(img_np)

    # 全局归一化
    all_activations = np.array(all_activations)
    vmin, vmax = np.min(all_activations), np.max(all_activations)

    # 处理每张图片
    for i, (before, after, imgname, original_image) in enumerate(
            zip(before_activations, after_activations, imgnames, original_images)):

        # 显示原始图像
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        axs[i, 0].imshow(original_image)
        axs[i, 0].axis('off')

        # 处理前后激活图
        for j, activation in enumerate([before, after], start=1):
            if activation is not None:
                avg_activation = torch.mean(activation, dim=1, keepdim=True)
                img_np = avg_activation.squeeze().float().cpu().numpy()
                img_np = np.maximum(img_np, 1e-6)
                img_np = np.log1p(img_np)
                img_np = (img_np - vmin) / (vmax - vmin + 1e-8)
                img_np = np.power(img_np, 0.5)


                #
                # mean_feat_flat = img_np.flatten()
                #
                # # 绘制直方图（默认分成 10 个 bins）
                # plt.hist(mean_feat_flat, bins=50, color='blue', alpha=0.7)
                # plt.xlabel('Value')
                # plt.ylabel('Frequency')
                # plt.title('Histogram of mean_feat (2160x3840)')
                # plt.grid(True, linestyle='--', alpha=0.5)
                # plt.show()
                # plt.savefig("ActMap/zhifangtu.png", bbox_inches='tight', dpi=120)
                #
                #
                im = axs[i, j].imshow(img_np, cmap='viridis', vmin=0, vmax=1)
                axs[i, j].axis('off')
            else:
                axs[i, j].axis('off')

        # 关闭颜色条位置
        axs[i, -1].axis('off')


    # 更精细地调整间距
    plt.subplots_adjust(
        left=0.07,  # 左边距
        right=0.92,  # 右边距（为颜色条留空间）
        bottom=0.22,  # 底部空间（为标签留空间）
        top=0.95,  # 顶部空间
        wspace=0.01,  # 进一步减小水平间距
        hspace=0.03 if num_images > 1 else 0.1  # 垂直间距，单图时稍大
    )
    # 创建颜色条
    cbar_ax = fig.add_axes([0.91, 0.22, 0.01, 0.73])
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Activation intensity')

    # 调整间距
    plt.subplots_adjust(wspace=0.01, hspace=-0.1)

    # 添加标签
    col_positions = []
    for j in range(3):
        bbox = axs[-1, j].get_position()
        col_center = (bbox.x0 + bbox.x1) / 2
        col_positions.append(col_center)

    labels = ['Image', 'Before', 'After']
    # 调整标签位置（向下移动一点）
    label_y_position = 0.19  # 从0.12调整为0.10
    for j in range(3):
        fig.text(col_positions[j], label_y_position, labels[j],
                 ha='center', va='center', fontsize=40)


    # 保存图片
    pdf_path = os.path.join(save_dir, 'last_conv_activations.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.close()


def normalize_to_positive(img_np):
    min_val = np.min(img_np)
    max_val = np.max(img_np)
    if min_val < 0:
        return (img_np - min_val) / (max_val - min_val)
    else:
        return img_np / max_val


def register_hooks(model):
    activations = {}

    def get_activation(name, prefix, is_egem=False, is_conv=False):
        def hook(model, input, output):
            if not is_egem and not is_conv and 'ISBlock' in name:
                try:
                    x_id, x_hw, x_w, x_h = torch.split(output, model.split_indexes, dim=1)
                    if prefix not in activations:
                        activations[prefix] = [x_id, x_hw, x_w, x_h, None]
                    else:
                        activations[prefix][:4] = [x_id, x_hw, x_w, x_h]
                except Exception as e:
                    print(f"Error processing ISBlock {prefix}: {e}")

            elif is_egem and 'EGEM' in name:
                try:
                    if prefix not in activations:
                        activations[prefix] = [None, None, None, None, output]
                    else:
                        activations[prefix][4] = output
                except Exception as e:
                    print(f"Error processing EGEM {prefix}: {e}")

            elif is_conv:  # 处理普通卷积层
                try:
                    activations[prefix] = output  # 直接存储整个输出张量
                except Exception as e:
                    print(f"Error processing conv layer {prefix}: {e}")

        return hook

    hooks = []
    target_layers = [
        ('layers.0.residual_group.blocks.0.EGEM', 'first_layer_first_block', True, False),
        ('layers.0.residual_group.blocks.0.ISBlock', 'first_layer_first_block', False, False),
        ('layers.3.residual_group.blocks.4.EGEM', 'last_layer_last_block', True, False),
        ('layers.3.residual_group.blocks.4.ISBlock', 'last_layer_last_block', False, False),

        ('layers.3.residual_group', 'before_last_conv', False, True),  # 最后一层卷积前的特征
        ('layers.3.conv', 'after_last_conv', False, True),  # 最后一层卷积后的特征

    ]

    for layer_name, prefix, is_egem, is_conv in target_layers:
        try:
            layer = dict([*model.named_modules()])[layer_name]
            handle = layer.register_forward_hook(get_activation(layer_name, prefix, is_egem, is_conv))
            hooks.append(handle)
        except KeyError as e:
            print(f"Layer not found: {layer_name}, error: {e}")

    return activations, hooks


# default = '../experiments/HLKNet_SR_x2_light_Div2K/models/net_g_510000.pth',
# default = 'ckpt/ISM579/net_g_505000.pth',

# 4倍超分模型
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,
                        default='../experiments/HLKNet_SR_x4_light_Div2K/models/net_g_510000.pth',
                        help='path to load your pre_trained model weights')
    parser.add_argument('--input', type=str, default='Urban100/LR_bicubic/X2', help='input test image folder')
    parser.add_argument('--output', type=str, default='./ActMap/olkb/579', help='output folder')
    parser.add_argument('--image_indices', type=int, nargs='+', default=[4,94],
                        help='indices of the images to process in the dataset (space-separated list)')
    args = parser.parse_args()
# defuault=[4,94]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    init_model = buildModel(4)
    checkpoint = torch.load(args.model_path, map_location=device)
    init_model.load_state_dict(checkpoint['params'], strict=True)
    init_model.eval().to(device)

    save_path = os.path.join(args.output, str(time.time()))
    os.makedirs(save_path, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.input, '*')))
    if not paths:
        print("No images found in the input directory.")
        return

    # 验证所有图像索引是否有效
    valid_indices = []
    for idx in args.image_indices:
        if 0 <= idx < len(paths):
            valid_indices.append(idx)
        else:
            print(f"Invalid image index {idx}. Skipping.")

    if not valid_indices:
        print(f"No valid image indices. Please choose indices between 0 and {len(paths) - 1}.")
        return

    # 为每个层前缀创建列表以存储多个图像的激活
    first_layer_activations = []
    last_layer_activations = []
    imgnames = []
    original_images = []

    # 在主函数中替换这部分代码：
    before_last_conv_activations = []
    after_last_conv_activations = []

    # 处理每个选定的图像
    for idx in valid_indices:
        path = paths[idx]
        imgname = os.path.splitext(os.path.basename(path))[0]
        imgnames.append(imgname)
        print(f'Testing: {imgname}')

        original_image = cv2.imread(path, cv2.IMREAD_COLOR)
        original_images.append(original_image)

        img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img = img.unsqueeze(0).to(device)

        try:
            activations, hooks = register_hooks(init_model)

            with torch.no_grad():
                output = init_model(img)

            # 移除钩子
            for hook in hooks:
                hook.remove()

            # 保存SR结果
            output = output.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, f'{imgname}_SR.png'), output)

            # 收集激活数据
            if 'first_layer_first_block' in activations:
                first_layer_activations.append(activations['first_layer_first_block'])
            if 'last_layer_last_block' in activations:
                last_layer_activations.append(activations['last_layer_last_block'])

            # 收集普通卷积层的激活数据
            # if 'before_layer3_conv' in activations:
            #     before_layer3_conv_activations.append(activations['before_layer3_conv'])
            # if 'layer3_conv' in activations:
            #     layer3_conv_activations.append(activations['layer3_conv'])

            # 在处理每个图像时收集激活数据
            if 'before_last_conv' in activations:
                before_last_conv_activations.append(activations['before_last_conv'])
            if 'after_last_conv' in activations:
                after_last_conv_activations.append(activations['after_last_conv'])

        except Exception as error:
            print(f'Error processing {imgname}: {error}')

    # 保存多图激活图
    if first_layer_activations:
        save_activation_mean_maps(first_layer_activations, imgnames, save_path, 'first_layer', original_images)
    if last_layer_activations:
        save_activation_mean_maps(last_layer_activations, imgnames, save_path, 'last_layer', original_images)

    # 保存普通卷积层的激活图
    # 最后保存激活图
    if before_last_conv_activations and after_last_conv_activations:
        save_before_after_activations(
            before_last_conv_activations,
            after_last_conv_activations,
            imgnames,
            save_path,
            original_images
        )



if __name__ == '__main__':
    main()
