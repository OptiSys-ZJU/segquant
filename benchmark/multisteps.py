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

def multistep_ctrl(attn_nums=6, max_steps=400, type='mse', quant_type='int8_smooth'):
    indices = list(range(max_steps))
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    plt.figure(figsize=(10, 5))
    input_errors = []
    timestamps = []
    for index in range(attn_nums):
        errors = []
        for i in indices:
            steps_lst = []
            #for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            for scale in [0.8]:
                fp16_all = torch.load(f'multistep/controlnet_output_fp16_canny_{scale}_{i}.pt')
                int8_all = torch.load(f'multistep/controlnet_output_{quant_type}_canny_{scale}_{i}.pt')

                if index == 0:
                    fp16 = fp16_all['hidden_states']
                    int8 = int8_all["hidden_states"]
                    input_errors.append(error(fp16, int8, type))
                    timestamps.append(fp16_all['timestep'][0].item())

                fp16 = fp16_all["control_block_samples"][index] / fp16_all['cond_scale']
                int8 = int8_all["control_block_samples"][index] / fp16_all['cond_scale']
                steps_lst.append(error(fp16, int8, type))
            this_error = sum(steps_lst) / len(steps_lst)
            # errors.append(math.log(this_error))
            errors.append(this_error)
        
        plt.plot(indices, errors, color=colors[index], label=f'Output{index}')
    
    plt.plot(indices, input_errors, color='black', label=f'Latent')
    plt.plot(indices, timestamps, color='gray', label=f'Timestamp(Value)')
    plt.xlabel('Step Index (i)')
    plt.ylabel('Average Diff Error')
    plt.title('Error Visualization Over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'stat_control_multisteps_{quant_type}_{type}.png', dpi=300, bbox_inches='tight')
    plt.close()

def multistep_dit(max_steps=200, type='mse', quant_type='int8_smooth', ctrl_type='canny'):
    input1 = []
    input_blocks = [[] for _ in range(6)]
    input2 = [[] for _ in range(6)]
    errors = []
    timestamps = []
    indices = list(range(max_steps))  # x 轴对应的 i 值

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    for i in indices:
        steps_lst = []
        for scale in [0.8]:
        # for scale in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            fp16_all = torch.load(f'multistep/dit_output_fp16_{ctrl_type}_{scale}_{i}.pt')
            int8_all = torch.load(f'multistep/dit_output_{quant_type}_{ctrl_type}_{scale}_{i}.pt')

            timestamps.append(fp16_all['timestep'][0].item())

            fp16 = fp16_all['hidden_states']
            int8 = int8_all["hidden_states"]
            input1.append(error(fp16, int8, type))

            for i in range(6):
                fp16 = fp16_all['block_res_samples'][i]
                int8 = int8_all["block_res_samples"][i]
                input_blocks[i].append(error(fp16, int8, type))

                fp16 = fp16_all['control_block_samples'][i]
                int8 = int8_all["control_block_samples"][i]
                input2[i].append(error(fp16, int8, type))

            fp16 = fp16_all['noise_pred']
            int8 = int8_all['noise_pred']
            steps_lst.append(error(fp16, int8, type))
        this_error = sum(steps_lst) / len(steps_lst)
        errors.append(this_error)
    
    # ========== 第一张图（折线图） ==========
    plt.figure(figsize=(10, 5))
    plt.plot(indices, input1, color='black', label='Latent')
    # plt.plot(indices, timestamps, color='gray', label='Timestamp(Value)')
    for i in range(6):
        pass
        # plt.plot(indices, input_blocks[i], linestyle='--', color=colors[i], label=f'ControlNet Block[{i}]')
        # plt.plot(indices, input2[i], linestyle='-', color=colors[i], label=f'ControlNet[{i}]')
    plt.plot(indices, errors, color='red', label='Noise Pred')
    plt.xlabel('Step Index (i)')
    plt.ylabel('Average Error')
    plt.title(f'{ctrl_type} Error Visualization Over Steps')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True)
    plt.savefig(f'stat_dit_multisteps_{quant_type}_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 第二张图（散点图） ==========
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 行 3 列

    # for i in range(6):
    #     row, col = divmod(i, 3)
    #     ax = axes[row, col]
    #     corr_coef, _ = pearsonr(timestamps, input2[i])
    #     ax.scatter(timestamps, input2[i], color=colors[i], alpha=0.6)
    #     ax.set_title(f'{ctrl_type} ControlNet[{i}] (Corr: {corr_coef:.2f})')
    #     ax.set_xlabel('Timestamp')
    #     ax.set_ylabel('Error')
    #     ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'stat_dit_multisteps_scatter_{quant_type}_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    plt.close()

    # plt.figure(figsize=(8, 5))
    # for i in range(max_steps):
    #     plt.plot(range(6), [input_blocks[j][i] for j in range(6)], color=colors[0], alpha=0.6)
    # plt.xlabel("Block Index")
    # plt.ylabel("Values")
    # plt.title("Scatter Plot of Input Blocks")
    # plt.grid(True, linestyle='--', alpha=0.5)
    # plt.savefig(f'stat_dit_multisteps_attn_{quant_type}_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--ctrl_types', nargs='+', help="List of Types", required=True)
    parser.add_argument('--quant_types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print('Ctrl types:', args.ctrl_types, "All types:", args.quant_types)
    for ctrl_type in args.ctrl_types:
        for quant_type in args.quant_types:
            multistep_dit(quant_type=quant_type, type='frobenius', ctrl_type=ctrl_type)
    # multistep_ctrl(quant_type=quant_type, type='frobenius')
    # multistep_dit(quant_type=quant_type, type='frobenius')