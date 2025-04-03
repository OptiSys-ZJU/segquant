import argparse
import numpy as np
from sklearn.decomposition import FastICA
import torch
import copy
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import shapiro
import scipy.stats as stats
from scipy.stats import skew, kurtosis
from scipy.stats import t

def minus_list(A, B):
    return [a - b for a, b in zip(A, B)]

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


def wrap_time_vec(quant_type, ctrl_type, scale, i):
    tsemb = torch.load(f'time_error/{quant_type}_{ctrl_type}_{scale}_tsemb_{i}.pt')
    adanorm = torch.load(f'time_error/{quant_type}_{ctrl_type}_{scale}_adanorm_{i}.pt')

    return {
        "input": tsemb['timesteps_proj'],
        "l1_output": tsemb['l1_output'],
        "silu1_output": tsemb['act_output'],
        "l2_output": tsemb['l2_output'],
        "silu2_output": adanorm['linear_input'],
        "l3_out": adanorm['linear_out'],
    }


def time_error_analysis(max_steps=200, type='mse', ctrl_type='canny'):
    indices = list(range(max_steps))  # x 轴对应的 i 值
    timesteps = []
    error_finals = []

    xs = []
    ys = []

    for i in indices:
        scale = 0.8
        timestep = torch.load(f'singlestep/fp16_{ctrl_type}_{scale}_{i}.pt')['timestep'][0].cpu().item()
        print(i, timestep)
        timesteps.append(timestep)

        fp16_all = wrap_time_vec('fp16', ctrl_type, scale, i)
        int8_enabletime = wrap_time_vec('int8_smooth_enabletime', ctrl_type, scale, i)
        

        # error_final = error(fp16_all['l3_out'], int8_enabletime['l3_out'], 'frobenius')
        # error_finals.append(error_final)
        mat_error_final = (fp16_all['silu2_output'] - int8_enabletime['silu2_output']).cpu()

        if i % 25 == 0:
            x = fp16_all['silu2_output'].cpu().flatten().numpy()
            y = mat_error_final.flatten().numpy()    
            plt.scatter(x, y, color='blue', label='Data Points', s=10, alpha=0.5)
            plt.title('Scatter Plot of xlist vs ylist')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.savefig(f'stat_timeerror_scat_{ctrl_type}_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()

    
    # plt.figure(figsize=(10, 5))
    # plt.plot(timesteps, error_finals, color='red', label='l3_out')

    # plt.xlabel('Timesteps')
    # plt.ylabel('Average Error')
    # plt.title(f'{ctrl_type} Error Visualization Single Step')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.savefig(f'stat_timeerror_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()



def singlestep_analysis(max_steps=200, type='mse', ctrl_type='canny'):
    real_inputs = []
    timesteps = []

    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']

    noise_correlation_mats = []
    ctrl_correlation_mats = [[] for _ in range(6)]
    
    ctrl_enabletime = [[] for _ in range(6)]
    ctrl_enablelatent = [[] for _ in range(6)]
    ctrl_enableall = [[] for _ in range(6)]


    noise_enabletime = []
    noise_enablelatent = []
    noise_enableall = []
    indices = list(range(max_steps))  # x 轴对应的 i 值

    for i in indices:
        scale = 0.8
        fp16_all = torch.load(f'singlestep/fp16_{ctrl_type}_{scale}_{i}.pt')
        int8_onlytime = torch.load(f'singlestep/int8_smooth_enabletime_{ctrl_type}_{scale}_{i}.pt')
        int8_onlylatent = torch.load(f'singlestep/int8_smooth_enablelatent_{ctrl_type}_{scale}_{i}.pt')
        int8_all = torch.load(f'singlestep/int8_smooth_{ctrl_type}_{scale}_{i}.pt')

        timestep = fp16_all['timestep'][0].cpu().item()
        print(timestep)
        timesteps.append(timestep)

        # real_inputs.append((fp16_all['hidden_states'], fp16_all['timestep']))

        # for i in range(6):
        #     fp16 = fp16_all['control_block_samples'][i]
        #     int8_1 = int8_onlytime["control_block_samples"][i]
        #     int8_2 = int8_onlylatent["control_block_samples"][i]
        #     int8_3 = int8_all["control_block_samples"][i]
        #     # mat_ctrl_enabletime = copy.deepcopy(fp16-int8_1)
        #     ctrl_enabletime[i].append(error(fp16, int8_1, type))

        #     # mat_ctrl_enablelatent = copy.deepcopy(fp16-int8_2)
        #     ctrl_enablelatent[i].append(error(fp16, int8_2, type))

        #     # mat_ctrl_enableall = copy.deepcopy(fp16-int8_3)
        #     ctrl_enableall[i].append(error(fp16, int8_3, type))

        fp16 = fp16_all['noise_pred']

        int8 = int8_onlytime['noise_pred']
        mat_noise_enabletime = fp16 - int8
        # noise_enabletime.append(error(fp16, int8, type))

        # int8 = int8_onlylatent['noise_pred']
        # # mat_noise_enablelatent = copy.deepcopy(fp16 - int8)
        # noise_enablelatent.append(error(fp16, int8, type))

        # int8 = int8_all['noise_pred']
        # mat_noise_enableall = copy.deepcopy(fp16 - int8)
        # noise_enableall.append(error(fp16, int8, type))


        if i % 50 == 24:
            x = fp16.cpu().flatten().numpy()
            y = mat_noise_enabletime.cpu().flatten().numpy()    
            plt.scatter(x, y, color='blue', label='Data Points', s=10, alpha=0.5)
            plt.title('Scatter Plot of xlist vs ylist')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.legend()
            plt.savefig(f'stat_time_scat_{ctrl_type}_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()

    ###########################################
    # plt.figure(figsize=(10, 5))
    # plt.plot(timesteps, noise_enabletime, color='red', label='enabletime')
    # plt.plot(timesteps, noise_enablelatent, color='green', label='enablelatent')
    # plt.plot(timesteps, minus_list(noise_enableall, noise_enabletime), color='blue', label='enableall')

    # plt.xlabel('Step Index (i)')
    # plt.ylabel('Average Error')
    # plt.title(f'{ctrl_type} Error Visualization Single Step')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.grid(True)
    # plt.savefig(f'stat_singlestep_noise_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # ###########################################
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 创建 1 行 2 列的子图
    # corr_coef1, _ = pearsonr(timesteps, noise_enabletime)
    # axes[0].scatter(timesteps, noise_enabletime, color=colors[0], alpha=0.6)
    # axes[0].set_xlabel('Timestep')
    # axes[0].set_ylabel('Average Error')
    # axes[0].set_title(f'{ctrl_type} enabletime Noise Error (Corr: {corr_coef1:.2f})')
    # axes[0].grid(True)
    # corr_coef2, _ = pearsonr(timesteps, noise_enablelatent)
    # axes[1].scatter(timesteps, noise_enablelatent, color=colors[1], alpha=0.6)
    # axes[1].set_xlabel('Timestep')
    # axes[1].set_ylabel('Other Error')
    # axes[1].set_title(f'{ctrl_type} enablelatent Other Error (Corr: {corr_coef2:.2f})')
    # axes[1].grid(True)
    # plt.tight_layout()
    # plt.savefig(f'stat_singlestep_scatter_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # ###########################################
    # # plt.figure(figsize=(6, 6))
    # # noise_mean = np.mean(noise_correlation_mats, axis=0)
    # # cax = plt.imshow(noise_mean, cmap='coolwarm', interpolation='none')
    # # print('noise corr:')
    # # print(noise_mean)
    # # plt.colorbar(cax)
    # # # plt.xticks(np.arange(3), ['Time', 'Latent', 'All'])
    # # # plt.yticks(np.arange(3), ['Time', 'Latent', 'All'])
    # # plt.xticks(np.arange(3), ['InputT', 'Time', 'Latent'])
    # # plt.yticks(np.arange(3), ['InputT', 'Time', 'Latent'])
    # # plt.title('Average Error Correlation Matrix')
    # # plt.grid(False)
    # # plt.savefig(f'stat_singlestep_noise_corr_{ctrl_type}.png', dpi=300, bbox_inches='tight')



    # ###########################################
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 创建2行3列的子图
    # fig.suptitle(f'{ctrl_type} Error Visualization Single Step', fontsize=16)
    # for i, ax in enumerate(axes.flat):
    #     if i < 6:
    #         ax.plot(timesteps, ctrl_enabletime[i], color='red', label='enabletime')
    #         ax.plot(timesteps, ctrl_enablelatent[i], color='green', label='enablelatent')
    #         ax.plot(timesteps, minus_list(ctrl_enableall[i], ctrl_enabletime[i]), color='blue', label='enableall')

    #         ax.set_xlabel('Timestep')
    #         ax.set_ylabel('Average Error')
    #         ax.set_title(f'AttnBlock {i}')
    #         ax.legend()
    #         ax.grid(True)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局避免重叠
    # plt.savefig(f'stat_singlestep_ctrl_{ctrl_type}.png', dpi=300, bbox_inches='tight')

    # ###########################################
    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 行 3 列

    # for i in range(6):
    #     row, col = divmod(i, 3)
    #     ax = axes[row, col]
    #     corr_coef, _ = pearsonr(timesteps, ctrl_enabletime[i])
    #     ax.scatter(timesteps, ctrl_enabletime[i], color=colors[i], alpha=0.6)
    #     ax.set_title(f'{ctrl_type} ControlNet[{i}] enableTime (Corr: {corr_coef:.2f})')
    #     ax.set_xlabel('Timestamp')
    #     ax.set_ylabel('Error')
    #     ax.grid(True)

    # plt.tight_layout()
    # plt.savefig(f'stat_singlestep_scatter_time_ctrl_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    # fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 行 3 列

    # for i in range(6):
    #     row, col = divmod(i, 3)
    #     ax = axes[row, col]
    #     corr_coef, _ = pearsonr(timesteps, ctrl_enablelatent[i])
    #     ax.scatter(timesteps, ctrl_enablelatent[i], color=colors[i], alpha=0.6)
    #     ax.set_title(f'{ctrl_type} ControlNet[{i}] enableLatent (Corr: {corr_coef:.2f})')
    #     ax.set_xlabel('Timestamp')
    #     ax.set_ylabel('Error')
    #     ax.grid(True)

    # plt.tight_layout()
    # plt.savefig(f'stat_singlestep_scatter_latent_ctrl_{ctrl_type}.png', dpi=300, bbox_inches='tight')
    # plt.close()

    ###########################################
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # axes = axes.flatten()
    # for i, corr_mat in enumerate(ctrl_correlation_mats):
    #     ax = axes[i]  # 获取当前的子图
    #     avg_corr_mat = np.mean(corr_mat, axis=0)  # 计算每个实验的平均相关矩阵
    #     cax = ax.imshow(avg_corr_mat, cmap='coolwarm', interpolation='none')  # 显示相关性矩阵
    #     print(f'output{i}')
    #     print(avg_corr_mat)
    #     ax.set_title(f'Output{i} Error Correlation')  # 添加标题
    #     ax.set_xticks(np.arange(3))
    #     ax.set_yticks(np.arange(3))
    #     # ax.set_xticklabels(['Time', 'Latent', 'All'])
    #     # ax.set_yticklabels(['Time', 'Latent', 'All'])

    #     ax.set_xticklabels(['InputT', 'Time', 'Latent'])
    #     ax.set_yticklabels(['InputT', 'Time', 'Latent'])
    #     fig.colorbar(cax, ax=ax, shrink=0.8)  # 添加色条
    # plt.tight_layout()
    # plt.savefig(f'stat_singlestep_ctrl_corr_{ctrl_type}.png', dpi=300, bbox_inches='tight')




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--ctrl_types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print('Ctrl types:', args.ctrl_types)
    for ctrl_type in args.ctrl_types:
        # time_error_analysis(type='frobenius', ctrl_type=ctrl_type)
        singlestep_analysis(type='frobenius', ctrl_type=ctrl_type)