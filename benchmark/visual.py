import torch
import glob
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
from audtorch.metrics.functional import pearsonr
import seaborn as sns

def draw(tensor_list, type, scale, num_inference_step, step):
    num_tensors = len(tensor_list)
    
    # 增大图像尺寸
    fig = plt.figure(figsize=(15, 10))  # 调整 figsize 以增加图像的大小

    # 固定 2 行 3 列布局
    rows = 2
    cols = 3

    for idx, tensor in enumerate(tensor_list):
        tensor_np = tensor.detach().cpu().numpy()
        ax = fig.add_subplot(rows, cols, idx + 1)
        cax = ax.imshow(tensor_np, cmap='viridis', interpolation='none')
        fig.colorbar(cax)
        ax.set_title(f'Tensor {idx+1}', fontsize=14)  # 增大标题字体
        ax.set_xlabel('X axis (Batch dimension)', fontsize=12)  # 增大轴标签字体
        ax.set_ylabel('Y axis (Channels dimension)', fontsize=12)  # 增大轴标签字体

    # 在整张图上添加 scale, num_inference_step, step 信息
    fig.suptitle(f"Scale: {scale}, Num Inference Step: {num_inference_step}, Step: {step}", fontsize=16)

    # 增大图像显示
    plt.tight_layout(pad=3.0)  # 增加子图之间的距离

    # 生成文件名
    filename = f"tensor_heatmap_scale-{type}-{scale}_steps-{num_inference_step}_step-{step}.png"
    save_path = os.path.join("visualizations", filename)  # 存入 'visualizations' 文件夹
    os.makedirs("visualizations", exist_ok=True)  # 确保目录存在

    plt.savefig(save_path, dpi=600)  # 增加 dpi 以提高分辨率
    print(f"Saved visualization to {save_path}")
    plt.close()

def frobenius_norm(tensor):
    return torch.norm(tensor, p='fro')

def analyze_tensor_distribution(tensor):
    # 统计正数、负数和零的数量
    num_positive = torch.sum(tensor > 0).item()
    num_negative = torch.sum(tensor < 0).item()
    num_zero = torch.sum(tensor == 0).item()

    total_elements = tensor.numel()  # 总元素数

    # 计算占比
    pos_ratio = num_positive / total_elements
    neg_ratio = num_negative / total_elements
    zero_ratio = num_zero / total_elements

    # 打印结果
    print(f"Total Elements: {total_elements}")
    print(f"Positive: {num_positive} ({pos_ratio:.2%})")
    print(f"Negative: {num_negative} ({neg_ratio:.2%})")
    print(f"Zero: {num_zero} ({zero_ratio:.2%})")

    return {
        "total": total_elements,
        "positive": num_positive,
        "negative": num_negative,
        "zero": num_zero,
        "positive_ratio": pos_ratio,
        "negative_ratio": neg_ratio,
        "zero_ratio": zero_ratio
    }

metadata = {}

def fit(x):
    return 37.48 * (x ** 2) - 20.47 * x + 147.7


def plot(folder_path):
    # for i in [1]:
    for i in range(0, 28, 1):
        diff_int8_list = []
        diff_int8disnorm1_list = []
        diff_int8block_list = []

        diff_fp8_list = []
        diff_fp8disnorm1_list = []
        diff_fp8block_list = []

        for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print(scale, i)
            fp16_input = torch.load(os.path.join(folder_path, f"norm1_input_fp16_canny_{scale}_{i}.pt"))
            fp16_output = torch.load(os.path.join(folder_path, f"norm1_output_fp16_canny_{scale}_{i}.pt"))
            # plot_tensor_distributions(fp16_input, fp16_output, f'fp16_{i}.png')

            int8dis_input = torch.load(os.path.join(folder_path, f"norm1_input_int8_smooth_disnorm1_canny_{scale}_{i}.pt"))
            int8dis_output = torch.load(os.path.join(folder_path, f"norm1_output_int8_smooth_disnorm1_canny_{scale}_{i}.pt"))
            # plot_tensor_distributions(int8dis_input, int8dis_output, f'int8_smooth_disnorm1_{i}.png')

            int8_input = torch.load(os.path.join(folder_path, f"norm1_input_int8_smooth_canny_{scale}_{i}.pt"))
            int8_output = torch.load(os.path.join(folder_path, f"norm1_output_int8_smooth_canny_{scale}_{i}.pt"))
            # plot_tensor_distributions(int8_input, int8_output, f'int8_smooth_{i}.png')

            int8_block_input = torch.load(os.path.join(folder_path, f"norm1_input_int8_smooth_block_canny_{scale}_{i}.pt"))
            int8_block_output = torch.load(os.path.join(folder_path, f"norm1_output_int8_smooth_block_canny_{scale}_{i}.pt"))

            fp8_output = torch.load(os.path.join(folder_path, f"norm1_output_fp8_canny_{scale}_{i}.pt"))
            fp8dis_output = torch.load(os.path.join(folder_path, f"norm1_output_fp8_disnorm1_canny_{scale}_{i}.pt"))
            fp8_block_output = torch.load(os.path.join(folder_path, f"norm1_output_fp8_block_canny_{scale}_{i}.pt"))

            diff_int8_list.append(fp16_output-int8_output)
            diff_int8disnorm1_list.append(fp16_output-int8dis_output)
            diff_int8block_list.append(fp16_output-int8_block_output)
            diff_fp8_list.append(fp16_output-fp8_output)
            diff_fp8disnorm1_list.append(fp16_output-fp8dis_output)
            diff_fp8block_list.append(fp16_output-fp8_block_output)

            err1 = frobenius_norm(fp16_output-int8_output)
            print('int8smooth', err1.item())

            err2 = frobenius_norm(fp16_output-int8dis_output)
            print('int8disnorm1', err2.item())

            err3 = frobenius_norm(fp16_output-int8_block_output)
            print('int8block', err3.item())

            err4 = frobenius_norm(fp16_output-fp8_output)
            print('fp8', err4.item())

            err5 = frobenius_norm(fp16_output-fp8dis_output)
            print('fp8disnorm1', err5.item())

            err6 = frobenius_norm(fp16_output-fp8_block_output)
            print('fp8block', err6.item())
        
        plot_diff(diff_int8_list, diff_int8disnorm1_list, diff_int8block_list, f'diff_int8_{i}.png')
        plot_diff(diff_fp8_list, diff_fp8disnorm1_list, diff_fp8block_list, f'diff_fp8_{i}.png')

def plot_diff(tensor1_list, tensor2_list, tensor3_list, save_path):
    plt.figure(figsize=(12, 6))  # 创建单个图
    
    colors = ['blue', 'red', '#2ca02c']  # 颜色定义
    linewidth = 0.3  # 适当加粗线条

    # 计算均值，保持原始形状 (2, 1536)
    tensor1_mean = torch.mean(torch.stack(tensor1_list), dim=0).detach().cpu().numpy()
    tensor2_mean = torch.mean(torch.stack(tensor2_list), dim=0).detach().cpu().numpy()
    tensor3_mean = torch.mean(torch.stack(tensor3_list), dim=0).detach().cpu().numpy()

    # 绘制两行数据
    for i in range(2):
        plt.plot(range(tensor1_mean.shape[1]), tensor1_mean[i], label=f'Tensor1 Mean Row {i}', color=colors[0], linewidth=linewidth, alpha=0.8)
        plt.plot(range(tensor2_mean.shape[1]), tensor2_mean[i], label=f'Tensor2 Mean Row {i}', color=colors[1], linewidth=linewidth, alpha=0.8)
        plt.plot(range(tensor3_mean.shape[1]), tensor3_mean[i], label=f'Tensor3 Mean Row {i}', color=colors[2], linewidth=linewidth, alpha=0.8)

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Comparison of Mean Tensors')
    plt.legend()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()


def plot_tensor_distributions(tensor1, tensor2, save_path="tensor_distributions.png"):
    tensor1_flat = tensor1.detach().cpu().numpy()
    tensor2_flat = tensor2.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 创建两个子图

    axes[0].plot(range(1536), tensor1_flat[0], label='Tensor1 Channel 0')
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Channel-wise Values of Tensor (2, 1536)')
    axes[0].legend()

    axes[1].plot(range(9216), tensor2_flat[0], label='Tensor2 Channel 0')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Channel-wise Values of Tensor (2, 9216)')
    axes[1].legend()

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')


def visual(folder_path, type, alpha):
    fp16_file_paths = sorted(glob.glob(os.path.join(folder_path, "norm1_fp16_canny_1_0.8_input.pt")))
    for filepath in fp16_file_paths:
        print(filepath)
        fp16_tensor_list = torch.load(filepath)[0]
        this_tensor_list = torch.load(filepath.replace('fp16', type))[0]

        _, _, _, _, scale, num_inference_step, this_step = os.path.basename(filepath).split('_')
        scale = float(scale)
        num_inference_step = int(num_inference_step)
        this_step = int(this_step.replace('.pt', ''))
        
        this_res_list = []
        this_max_list = []
        this_norm_diff_list = []
        this_orig_max_list = []
        this_orig_norm_list = []
        new_tensors = []

        scale_k = []
        for i in range(len(fp16_tensor_list)):
            scale_k.append(fit(i))
            # scale_k.append(1)

        # print(scale_k)
        # exit(0)

        for i in range(len(fp16_tensor_list)):
            fp16_tensor = fp16_tensor_list[i]
            fp16_norm_tensor = fp16_tensor / scale / scale_k[i]
            this_tensor = this_tensor_list[i]
            this_norm_tensor = this_tensor / scale / scale_k[i]
            norm_diff = fp16_norm_tensor - this_norm_tensor

            # norm_diff = torch.load(f'perfect/{type}_0.9_{this_step}_{i}.pt')
            # torch.save(norm_diff, f'{type}_{scale}_{this_step}_{i}.pt')
            this_norm_diff_list.append(norm_diff)

        for i in range(len(fp16_tensor_list)):
            fp16_tensor = fp16_tensor_list[i]
            fp16_norm_tensor = fp16_tensor / scale / scale_k[i]

            this_tensor = this_tensor_list[i]
            this_norm_tensor = this_tensor / scale / scale_k[i]

            #analyze_tensor_distribution(standard_diff_i - standard_diff_0)

            new_tensor = (this_norm_tensor + this_norm_diff_list[i]) * scale * scale_k[i]
            new_tensors.append(new_tensor)

            f_norm = frobenius_norm(fp16_tensor - this_tensor)
            this_orig_max_list.append(f_norm.item())

            f_norm = frobenius_norm(fp16_norm_tensor - this_norm_tensor)
            this_orig_norm_list.append(f_norm.item())

            f_norm = frobenius_norm(fp16_tensor - new_tensor)
            this_max_list.append(f_norm.item())

        if scale not in metadata:
            metadata[scale] = {}
        if num_inference_step not in metadata[scale]:
            metadata[scale][num_inference_step] = {}
        
        metadata[scale][num_inference_step][this_step] = this_max_list
        print(filepath, alpha, 'ok')
        print('orig  ', this_orig_max_list)
        print('orig-n', this_orig_norm_list)
        print('new   ', this_max_list)
        print('---------------------------')
        # draw(this_res_list, type, scale, num_inference_step, this_step)

        # exit(0)
        

            

if __name__ == '__main__':
    # visual('latent_dump_output/controlnet', 'int8default', 0)
    # print('===============================')
    # visual('latent_dump_output/controlnet', 'int8smooth', 0)
    # visual('latent_dump_output/controlnet', 'int8smooth', 0)

    plot('norm1')
    # for alpha in np.arange(0, 1, 0.01):
    #     visual('latent_dump_output/controlnet', 'int8default', alpha)

    # visual('latent_dump_output/controlnet', 'int8smooth')
    # for scale in metadata:
    #     for num_inference_step in metadata[scale]:
    #         sorted_data = {k: metadata[scale][num_inference_step][k] for k in sorted(metadata[scale][num_inference_step].keys(), key=lambda x: int(x))}
    #         metadata[scale][num_inference_step] = sorted_data

    # with open("int8-default-scale.json", "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, indent=4, ensure_ascii=False)