import torch
import matplotlib.pyplot as plt
import math
import argparse
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import rbf_kernel
from scipy import stats
import numpy as np
from fitter import Fitter
from distfit import distfit
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def mae(A, B):
    return torch.mean(torch.abs(A - B))

def mse(A, B):
    return torch.mean((A - B) ** 2)

def frobenius(A, B):
    return torch.norm(A - B, p='fro')

def psnr(A, B):
    mse_val = mse(A, B)
    max_val = torch.max(A)
    return 10 * torch.log10(max_val**2 / mse_val)

def error(A, B, type):
    return globals()[type](A, B).item()



def gaussian_kernel(x, y, sigma=1.0):
    # 确保输入数据类型是 float32
    x = x.float()  # 转换为 float32
    y = y.float()  # 转换为 float32

    # 计算两组样本之间的欧几里得距离
    dist = torch.cdist(x, y, p=2) ** 2  # 计算平方欧几里得距离
    return torch.exp(-dist / (2 * sigma ** 2))  # 计算高斯核

# 计算 MMD 的函数
def compute_mmd(A, B, sigma=1.0):
    # 扁平化每个样本 [16, 128, 128] -> [16, 128*128]
    A_flat = A.view(A.size(0), -1)  # 形状变为 [16, 128*128]
    B_flat = B.view(B.size(0), -1)  # 形状变为 [16, 128*128]
    
    # 计算 A 内部的核矩阵 K_AA
    K_AA = gaussian_kernel(A_flat, A_flat, sigma)  # 计算 A 中每对样本的核
    K_AA = K_AA.sum() / (A.size(0) ** 2)  # 核矩阵的平均值

    # 计算 B 内部的核矩阵 K_BB
    K_BB = gaussian_kernel(B_flat, B_flat, sigma)  # 计算 B 中每对样本的核
    K_BB = K_BB.sum() / (B.size(0) ** 2)  # 核矩阵的平均值

    # 计算 A 和 B 之间的核矩阵 K_AB
    K_AB = gaussian_kernel(A_flat, B_flat, sigma)  # 计算 A 和 B 之间每对样本的核
    K_AB = K_AB.sum() / (A.size(0) * B.size(0))  # 核矩阵的平均值

    # 计算 MMD 的平方
    mmd_squared = K_AA + K_BB - 2 * K_AB
    return mmd_squared



def multi_single(max_steps=200, type='mse', quant_type='int8_smooth', ctrl_type='canny'):
    indices = list(range(max_steps))  # x 轴对应的 i 值
    timestamps = []

    singles = []
    multis = []

    for i in indices:
        for scale in [0.8]:
            fp16_all_multi = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all_multi = torch.load(f'multistep/{quant_type}_{ctrl_type}_{scale}_{i}.pt')

            timestamp = fp16_all_multi['timestep'][0].item()
            timestamps.append(timestamp)

            # error_multi = error(fp16_all_multi['noise_pred'], int8_all_multi['noise_pred'], type)
            mat_error_multi = fp16_all_multi['noise_pred'] - int8_all_multi['noise_pred']
            multi_vec = mat_error_multi.cpu().detach().flatten().numpy()
            multis.append(multi_vec)

            fp16_all_single = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all_single = torch.load(f'singlestep/{quant_type}_{ctrl_type}_{scale}_{i}.pt')
            
            
            # error_single = error(fp16_all_single['noise_pred'], int8_all_single['noise_pred'], type)
            mat_error_single = fp16_all_single['noise_pred'] - int8_all_single['noise_pred']
            single_vec = mat_error_single.cpu().detach().flatten().numpy()
            singles.append(single_vec)



    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, singles, color='red', label='singles')
    # plt.plot(timestamps, multis, color='blue', label='multis')

    # plt.xlabel('Timestamps')
    # plt.ylabel('Average Error')
    # plt.title(f'{ctrl_type} Error Visualization Single Step')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.savefig(f'stat_noise_sm_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

def find_extreme_values(tensor, n):
    # 计算每个元素与均值的差距
    mean = tensor.mean()
    abs_diff = torch.abs(tensor - mean)
    
    # 展平 tensor 并按差距排序
    flat_diff = abs_diff.view(-1)
    _, topk_indices = torch.topk(flat_diff, n, largest=True)
    
    # 获取对应的索引
    extreme_indices = torch.unravel_index(topk_indices, tensor.shape)
    
    return extreme_indices
            
def draw_hist(tensor, quant_type, i):
    noise_pred_uncond, noise_pred_text = tensor.chunk(2)
    mean_uncond = noise_pred_uncond.mean().item()
    std_uncond = noise_pred_uncond.std().item()
    mean_text = noise_pred_text.mean().item()
    std_text = noise_pred_text.std().item()

    # # 创建子图
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # # 绘制 noise_pred_uncond 的直方图
    # axes[0].hist(noise_pred_uncond.flatten().cpu().numpy(), bins=50, color='blue', alpha=0.7)
    # axes[0].set_title(f'Step{i} Histogram of noise_pred_uncond')
    # axes[0].axvline(mean_uncond, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_uncond:.2f}')
    # axes[0].axvline(mean_uncond + std_uncond, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_uncond:.2f}')
    # axes[0].legend()

    # # 绘制 noise_pred_text 的直方图
    # axes[1].hist(noise_pred_text.flatten().cpu().numpy(), bins=50, color='orange', alpha=0.7)
    # axes[1].set_title(f'Step{i} Histogram of noise_pred_text')
    # axes[1].axvline(mean_text, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_text:.2f}')
    # axes[1].axvline(mean_text + std_text, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_text:.2f}')
    # axes[1].legend()

    # # 调整布局并保存图像
    # plt.tight_layout()
    # plt.savefig(f'noise_pred_{quant_type}_{i}.png', dpi=300, bbox_inches='tight')

def hidden_ctrl(max_steps=200, quant_type='int8_smooth', ctrl_type='canny'):
    indices = list(range(max_steps))  # x 轴对应的 i 值
    for i in indices:
        for scale in [0.8]:
            fp16_all = torch.load(f'ctrl_hidden/fp16_{ctrl_type}_{scale}_{i}.pt')
            # int8_all = torch.load(f'ctrl_hidden/{quant_type}_{ctrl_type}_{scale}_{i}.pt')

            if i in [0, 199]:
                keys = ['init_hiddens', 'hidden_before', 'controlnet_cond', 'cond_output', 'hidden_states']
                titles = ['Init', 'Before', 'CondInit', 'Cond', 'After']
                fig, axes = plt.subplots(1, 5, figsize=(25, 5))

                for idx, key in enumerate(keys):
                    
                    data = fp16_all[key].cpu()[0].flatten().numpy().astype(np.float32)

                    print(keys[idx])
                    print('max: ', np.max(data))
                    print('min: ', np.min(data))
                    print('mean: ', np.mean(data))
                    print('std: ', np.std(data))

                    vmin, vmax = np.min(data), np.max(data)
                    axes[idx].hist(data, bins=50, color='blue', alpha=0.7)
                    axes[idx].set_title(f'{titles[idx]}')
                    axes[idx].set_xlim(vmin, vmax)
                    axes[idx].set_xticks(np.linspace(vmin, vmax, num=6))
                    axes[idx].axvline(np.mean(data), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(data):.2f}')
                    axes[idx].axvline(np.mean(data)+np.std(data), color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {np.std(data):.2f}')

                print('------')
                plt.tight_layout()
                plt.savefig(f'stat_ctrl_{ctrl_type}_{i}.png', dpi=300, bbox_inches='tight')

            


def multistep_dit(max_steps=200, type='mse', quant_type='int8_smooth', ctrl_type='canny'):
    input1 = []
    input_blocks = [[] for _ in range(6)]
    input2 = [[] for _ in range(6)]
    errors = []
    timestamps = []
    indices = list(range(max_steps))  # x 轴对应的 i 值

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    df_values = []
    loc_values = []
    scale_values = []
    last_mat = None

    mats = []
    means = []
    stds = []
    for i in indices:
        for scale in [0.8]:
            fp16_all = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{max_steps}_{i}.pt')
            int8_all = torch.load(f'multistep/{quant_type}_{ctrl_type}_{scale}_{max_steps}_{i}.pt')

            timestamps.append(fp16_all['timestep'][0].item())

            fp16 = fp16_all['noise_pred'].cpu()
            int8 = int8_all['noise_pred'].cpu()

            # fp16_noise_pred_uncond, fp16_noise_pred_text = fp16.chunk(2)
            # int8_noise_pred_uncond, int8_noise_pred_text = int8.chunk(2)

            # err_mat_uncond = fp16_noise_pred_uncond[0] - int8_noise_pred_uncond[0]
            # if (i + 1) % 25 == 0 or i == 0:
            #     draw_hist(fp16-int8, 'err', i)

            tensor = fp16-int8
            noise_pred_uncond, noise_pred_text = tensor.chunk(2)
            noise_pred_uncond = noise_pred_uncond[0]
            mean_uncond = noise_pred_uncond.mean().item()
            std_uncond = noise_pred_uncond.std().item()


            if timestamps[-1] >= 0 and timestamps[-1] <= 999:
                n_extremes = 2621  # 你可以调整这个数量
                extreme_indices_A = find_extreme_values(last_mat, n_extremes)
                extreme_indices_B = find_extreme_values(noise_pred_uncond, n_extremes)

                # # 3. 找出 A 中极端值的索引在 B 中也为极端值的索引
                # matching_indices = []
                # for idx_A in zip(*extreme_indices_A):
                #     if idx_A in zip(*extreme_indices_B):
                #         matching_indices.append(idx_A)

                # # 计算比例
                # matching_count = len(matching_indices)  # A 和 B 中都为极端值的索引数量
                # proportion = matching_count / n_extremes if n_extremes != 0 else 0  # 计算比例


                # 提取极端值索引并转换为 Python 原生 tuple（哈希友好）
                extreme_indices_A = set([tuple(int(x.item()) for x in idx) for idx in zip(*find_extreme_values(last_mat, n_extremes))])
                extreme_indices_B = set([tuple(int(x.item()) for x in idx) for idx in zip(*find_extreme_values(noise_pred_uncond, n_extremes))])

                # 使用集合交集加速匹配
                matching_indices = extreme_indices_A & extreme_indices_B
                matching_count = len(matching_indices)

                proportion = matching_count / n_extremes if n_extremes != 0 else 0

                # 输出结果
                print(i, timestamps[-1], matching_count, n_extremes, f"与上一时刻比例: {proportion:.4f}")
            
            last_mat = noise_pred_uncond

            # err_mat_uncond = fp16_noise_pred_uncond - int8_noise_pred_uncond
            # err_mat_text = fp16_noise_pred_text - int8_noise_pred_text
            # if (i + 1) % 25 == 0 or i == 0:
            #     flattened_data = noise_pred_uncond.view(-1).numpy()
            #     params = stats.t.fit(flattened_data)
            #     df, loc, scale = params
            #     print(df, loc, scale)
            
            # flattened_data = noise_pred_uncond.view(-1).numpy()
            # params = stats.t.fit(flattened_data)
            # df, loc, scale = params
            # df_values.append(df)
            # loc_values.append(loc)
            # scale_values.append(scale)


    # fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # # 绘制 df 参数
    # axes[0].plot(df_values, label='Degrees of Freedom (df)')
    # axes[0].set_title('Degrees of Freedom (df)')
    # axes[0].set_xlabel('Fitting Iteration')
    # axes[0].set_ylabel('df')
    # axes[0].legend()

    # # 绘制 loc 参数
    # axes[1].plot(loc_values, label='Location (loc)', color='orange')
    # axes[1].set_title('Location (loc)')
    # axes[1].set_xlabel('Fitting Iteration')
    # axes[1].set_ylabel('loc')
    # axes[1].legend()

    # # 绘制 scale 参数
    # axes[2].plot(scale_values, label='Scale (scale)', color='green')
    # axes[2].set_title('Scale (scale)')
    # axes[2].set_xlabel('Fitting Iteration')
    # axes[2].set_ylabel('scale')
    # axes[2].legend()
    # plt.tight_layout()
    # plt.savefig(f'stat_noise_t_{ctrl_type}.png', dpi=300, bbox_inches='tight')


    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, means, color='red', label='singles')

    # plt.xlabel('Timestamps')
    # plt.ylabel('Average Error')
    # plt.title(f'{ctrl_type} Error Visualization Single Step')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.savefig(f'stat_noise_mean_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # plt.figure(figsize=(10, 5))
    # plt.plot(timestamps, stds, color='red', label='singles')

    # plt.xlabel('Timestamps')
    # plt.ylabel('Average Error')
    # plt.title(f'{ctrl_type} Error Visualization Single Step')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.savefig(f'stat_noise_std_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()
    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--ctrl_types', nargs='+', help="List of Types", required=True)
    parser.add_argument('--quant_types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print('Ctrl types:', args.ctrl_types, "All types:", args.quant_types)
    for ctrl_type in args.ctrl_types:
        for quant_type in args.quant_types:
            hidden_ctrl(max_steps=200, quant_type=quant_type, ctrl_type=ctrl_type)
            # multistep_dit(max_steps=30, quant_type=quant_type, type='frobenius', ctrl_type=ctrl_type)
    # multistep_ctrl(quant_type=quant_type, type='frobenius')
    # multistep_dit(quant_type=quant_type, type='frobenius')