import numpy as np
from sklearn.decomposition import FastICA
import torch
import matplotlib.pyplot as plt
import math
import argparse
from scipy.stats import pearsonr


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

def gaussian_kernel(x, y, sigma=1.0):
    pairwise_dist = torch.cdist(x.unsqueeze(1), y.unsqueeze(0), p=2) ** 2
    return torch.exp(-pairwise_dist / (2 * sigma ** 2))

def mmd(A, B, sigma=1.0):
    Kaa = gaussian_kernel(A, A, sigma).mean()
    Kbb = gaussian_kernel(B, B, sigma).mean()
    Kab = gaussian_kernel(A, B, sigma).mean()
    return Kaa + Kbb - 2 * Kab

def error(A, B, type):
    return globals()[type](A, B).item()

def multistep_dit(max_steps=200, type='mse', ctrl_type='canny'):
    input1 = [[] for _ in range(6)]
    input2 = [[] for _ in range(6)]
    input3 = [[] for _ in range(6)]
    errors = []
    indices = list(range(max_steps))  # x 轴对应的 i 值

    for i in indices:
        steps_lst = []
        for scale in [0.8]:
        # for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            fp16_all = torch.load(f'multistep/dit_output_fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_onlytime = torch.load(f'multistep/dit_output_int8_smooth_disablelatent_{ctrl_type}_{scale}_{i}.pt')
            int8_onlylatent = torch.load(f'multistep/dit_output_int8_smooth_disabletime_{ctrl_type}_{scale}_{i}.pt')
            int8_all = torch.load(f'multistep/dit_output_int8_smooth_{ctrl_type}_{scale}_{i}.pt')

            for i in range(6):
                fp16 = fp16_all['control_block_samples'][i]
                int8_1 = int8_onlytime["control_block_samples"][i]
                int8_2 = int8_onlylatent["control_block_samples"][i]
                int8_3 = int8_all["control_block_samples"][i]
                input1[i].append(error(fp16, int8_1, type))
                input2[i].append(error(fp16, int8_2, type))
                input3[i].append(error(fp16, int8_3, type))

            fp16 = fp16_all['noise_pred']
            int8 = int8_all['noise_pred']
            steps_lst.append(error(fp16, int8, type))
        this_error = sum(steps_lst) / len(steps_lst)
        errors.append(this_error)

    for i in range(6):
        print(i)
        X = np.array([
            input1[i],
            input2[i],
            input3[i],
        ])

        np.set_printoptions(precision=6, suppress=True)
        ica = FastICA(n_components=3, random_state=0)
        S_ = ica.fit_transform(X.T)  # S_ 是独立误差分量
        W_ = ica.mixing_             # W_ 是混合矩阵

        # 输出分解结果
        # print("分离出的独立误差成分 (S):")
        # print(S_.T)

        print("\n混合矩阵 (W):")
        print(W_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--ctrl_types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print('Ctrl types:', args.ctrl_types)
    for ctrl_type in args.ctrl_types:
        multistep_dit(type='frobenius', ctrl_type=ctrl_type)