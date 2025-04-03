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




            


def multistep_dit(max_steps=200, type='mse', quant_type='int8_smooth', ctrl_type='canny'):
    input1 = []
    input_blocks = [[] for _ in range(6)]
    input2 = [[] for _ in range(6)]
    errors = []
    timestamps = []
    indices = list(range(max_steps))  # x 轴对应的 i 值

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    for i in indices:
        for scale in [0.8]:
            fp16_all = torch.load(f'multistep/fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all = torch.load(f'multistep/{quant_type}_{ctrl_type}_{scale}_{i}.pt')

            timestamps.append(fp16_all['timestep'][0].item())

            # fp16 = fp16_all['hidden_states']
            # int8 = int8_all["hidden_states"]
            # input1.append(error(fp16, int8, type))

            # for i in range(6):
            #     fp16 = fp16_all['block_res_samples'][i]
            #     int8 = int8_all["block_res_samples"][i]
            #     input_blocks[i].append(error(fp16, int8, type))

            #     fp16 = fp16_all['control_block_samples'][i]
            #     int8 = int8_all["control_block_samples"][i]
            #     input2[i].append(error(fp16, int8, type))

            fp16 = fp16_all['noise_pred']
            int8 = int8_all['noise_pred']

            mat_error_final = fp16 - int8
            if i % 25 == 24:
                x = fp16.cpu().flatten().numpy()
                y = mat_error_final.cpu().flatten().numpy()    
                plt.scatter(x, y, color='blue', label='Data Points', s=10, alpha=0.5)
                plt.title('Scatter Plot of xlist vs ylist')
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
                plt.legend()
                plt.savefig(f'stat_multi_{ctrl_type}_{i}.png', dpi=300, bbox_inches='tight')
                plt.close()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--ctrl_types', nargs='+', help="List of Types", required=True)
    parser.add_argument('--quant_types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print('Ctrl types:', args.ctrl_types, "All types:", args.quant_types)
    for ctrl_type in args.ctrl_types:
        for quant_type in args.quant_types:
            multi_single(quant_type=quant_type, type='frobenius', ctrl_type=ctrl_type)
    # multistep_ctrl(quant_type=quant_type, type='frobenius')
    # multistep_dit(quant_type=quant_type, type='frobenius')