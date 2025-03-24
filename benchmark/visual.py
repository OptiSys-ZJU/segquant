import torch
import glob
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from collections import defaultdict
from audtorch.metrics.functional import pearsonr

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

def visual(folder_path, type, alpha):
    fp16_file_paths = sorted(glob.glob(os.path.join(folder_path, "controlnet_output_fp16_canny_*_28_0.pt")))
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
    visual('latent_dump_output/controlnet', 'int8smooth', 0)
    # for alpha in np.arange(0, 1, 0.01):
    #     visual('latent_dump_output/controlnet', 'int8default', alpha)

    # visual('latent_dump_output/controlnet', 'int8smooth')
    # for scale in metadata:
    #     for num_inference_step in metadata[scale]:
    #         sorted_data = {k: metadata[scale][num_inference_step][k] for k in sorted(metadata[scale][num_inference_step].keys(), key=lambda x: int(x))}
    #         metadata[scale][num_inference_step] = sorted_data

    # with open("int8-default-scale.json", "w", encoding="utf-8") as f:
    #     json.dump(metadata, f, indent=4, ensure_ascii=False)