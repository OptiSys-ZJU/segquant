import torch

from segquant.torch.affiner import process_affiner


if __name__ == '__main__':
    from backend.torch.utils import randn_tensor
    from dataset.coco.coco_dataset import COCODataset
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel

    latents = randn_tensor((1, 16, 128, 128,), device=torch.device('cuda:0'), dtype=torch.float16)
    dataset = COCODataset(path='../dataset/controlnet_datasets/coco_canny', cache_size=16)

    model_quant = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')
    model_quant.transformer = torch.load('benchmark_record/run_seg_module/model/dit/model_quant_seg.pt', weights_only=False)
    model_real = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), 'cpu')

    config = {
        "solver": {
            "type": 'mserel',
            "blocksize": 128,
            "alpha": 0.5,
            "lambda1": 0.1,
            "lambda2": 0.1,
            "sample_mode": 'block',
            "percentile": 100,
            "greater": True,
            "scale": 1,
            'verbose': True,
        },

        "stepper": {
            'max_timestep': 30,
            'sample_size': 1,
            'recurrent': False,
            'noise_target': 'all',
            'enable_latent_affine': True,
            'enable_timesteps': None,
        },

        'extra_args': {
            "controlnet_conditioning_scale": 0,
            "guidance_scale": 7,
        }
    }

    affiner = process_affiner(config, dataset, model_real, model_quant, latents=latents, shuffle=False)

    # tmp
    real_latents = affiner.latent_record[0]
    quant_latents = affiner.latent_record[1]

    def compute_k_matrix(xs):
        sigmas = torch.tensor([s for s, _ in xs]).float()  # shape (N,)
        samples = torch.stack([s for _, s in xs])          # shape (N, ...)
        
        N, *shape = samples.shape
        samples_flat = samples.view(N, -1).to(dtype=torch.float32)  # shape (N, M)

        # 构建设计矩阵 X：shape (N, 2)，第一列是 sigma，第二列是常数项 1
        X = torch.stack([sigmas, torch.ones_like(sigmas)], dim=1)  # shape (N, 2)
        X = X.to(device=samples_flat.device, dtype=torch.float32)

        # 最小二乘解：theta: (2, M)
        result = torch.linalg.lstsq(X, samples_flat)
        theta = result.solution                             # shape (2, M)

        k_flat = theta[0]                                   # 斜率项
        k_matrix = k_flat.view(*shape)                      # reshape 回原 sample 形状
        return k_matrix


    real_perfect_v = compute_k_matrix(real_latents)
    quant_perfect_v = compute_k_matrix(quant_latents)

    def compute_signed_angle_matrix(real_ks: torch.Tensor, quant_ks: torch.Tensor, eps=1e-8):
        """
        计算 real_ks 相对 quant_ks 的带符号夹角（弧度）。
        形状假设是 (1, C, H, W)。

        返回：
            theta: torch.Tensor, 形状同输入，单位为度，范围 -180 到 180
        """
        k1 = real_ks
        k2 = quant_ks

        numerator = k1 - k2
        denominator = 1 + k1 * k2

        theta_rad = torch.atan(numerator / (denominator + eps))  # 夹角弧度，带符号
        theta_deg = theta_rad * (180.0 / torch.pi)              # 转成角度

        return theta_deg

    import matplotlib.pyplot as plt
    def visualize_k_heatmaps_4x4(real_ks: torch.Tensor, quant_ks: torch.Tensor, save_path="k_heatmaps_4x4.png"):
        """
        可视化两个 k 矩阵（real 和 quant）在每个通道上的差异，4x4 子图展示 real 和 quant 的对比。

        Args:
            real_ks: torch.Tensor, shape (1, 16, H, W)
            quant_ks: torch.Tensor, shape (1, 16, H, W)
            save_path: str, 图像保存路径
        """
        assert real_ks.shape == quant_ks.shape, "形状必须一致"
        assert real_ks.shape[1] == 16, "当前函数只支持 16 个通道"

        real_ks = real_ks.squeeze(0).cpu()   # (16, H, W)
        quant_ks = quant_ks.squeeze(0).cpu()

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i in range(16):
            row, col = divmod(i, 4)
            ax = axes[row, col]

            im = ax.imshow(real_ks[i], cmap='viridis', alpha=0.6, label='real')
            ax.imshow(quant_ks[i], cmap='plasma', alpha=0.4)

            ax.set_title(f"Channel {i}")
            ax.axis('off')

        fig.suptitle("real_ks (viridis) vs quant_ks (plasma)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path)
        plt.close()

    angles = compute_signed_angle_matrix(real_perfect_v, quant_perfect_v)
    def threshold_signed_angle_matrix(angles: torch.Tensor, threshold_degrees: float):
        """
        将小于指定角度阈值的 signed angle 置为 0。

        Args:
            angles: torch.Tensor，任意形状，角度单位为“度”
            threshold_degrees: float，阈值角度，绝对值小于该角度的将置为 0

        Returns:
            torch.Tensor，同 shape，应用了阈值过滤
        """
        mask = angles.abs() < threshold_degrees
        angles = angles.clone()  # 避免原地修改
        angles[mask] = 0.0
        return angles

    threshold_deg = 0.05
    angles = threshold_signed_angle_matrix(angles, threshold_deg)
    
    def plot_pos_neg_heatmap(angle_tensor: torch.Tensor, save_path="pos_neg_heatmap.png"):
        """
        输入角度矩阵，单位度，形状(1, C, H, W)
        生成只有1，0，-1的矩阵，1为正角度，-1为负角度，0为接近0
        并绘制16通道的4x4热力图
        
        Args:
            angle_tensor: torch.Tensor, shape (1, C, H, W), 单位为度
            save_path: str, 图片保存路径
        """
        # 去batch维度
        angles = angle_tensor.squeeze(0)
        C = angles.shape[0]
        assert C >= 16, "通道数不足16"
        
        # 生成标记矩阵
        pos_neg = torch.zeros_like(angles)
        pos_neg[angles > 1e-3] = 1
        pos_neg[angles < -1e-3] = -1
        # 介于[-1e-3, 1e-3] 的自动是0了
        
        fig, axs = plt.subplots(4, 4, figsize=(12, 12))
        axs = axs.flatten()
        
        for i in range(16):
            ax = axs[i]
            im = ax.imshow(pos_neg[i].cpu().numpy(), cmap='bwr', vmin=-1, vmax=1)
            ax.set_title(f"Channel {i}")
            ax.axis('off')
        
        # # 右侧加色条，只画一次即可
        # cbar = fig.colorbar(im, ax=axs.tolist(), orientation='vertical', fraction=0.02, pad=0.04, ticks=[-1,0,1])
        # cbar.ax.set_yticklabels(['-1 (neg)', '0', '1 (pos)'])
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    # plot_pos_neg_heatmap(angles)

    def channelwise_corr_and_fit(real_tensor: torch.Tensor, quant_tensor: torch.Tensor):
        """
        计算两个形状为 (1, C, H, W) 的 tensor 之间每个通道的：
            - Pearson 相关系数
            - 线性拟合函数系数 y = ax + b

        Args:
            real_tensor: torch.Tensor, shape (1, C, H, W)
            quant_tensor: torch.Tensor, shape (1, C, H, W)

        Returns:
            corrs: torch.Tensor, shape (C,)  每个通道的 Pearson 相关系数
            a: torch.Tensor, shape (C,)      每个通道线性拟合的斜率
            b: torch.Tensor, shape (C,)      每个通道线性拟合的截距
        """
        assert real_tensor.shape == quant_tensor.shape, "Tensor shapes must match"
        assert real_tensor.dim() == 4 and real_tensor.shape[0] == 1, "Expected shape (1, C, H, W)"
        
        # 去掉 batch 维度并展平空间维度
        real_flat = real_tensor.squeeze(0).view(real_tensor.shape[1], -1)   # (C, H*W)
        quant_flat = quant_tensor.squeeze(0).view(quant_tensor.shape[1], -1)

        x_mean = real_flat.mean(dim=1, keepdim=True)
        y_mean = quant_flat.mean(dim=1, keepdim=True)

        xm = real_flat - x_mean
        ym = quant_flat - y_mean

        # Pearson correlation
        corrs = (xm * ym).sum(dim=1) / (xm.norm(dim=1) * ym.norm(dim=1) + 1e-8)

        # Linear regression: y = a * x + b
        a = (xm * ym).sum(dim=1) / (xm ** 2).sum(dim=1)
        b = y_mean.squeeze(1) - a * x_mean.squeeze(1)

        return corrs, a, b

    corrs, a, b = channelwise_corr_and_fit(affiner.latent_record[1][-1][1], real_perfect_v)

    for i in range(len(corrs)):
        print(f"Channel {i:2d} | Corr: {corrs[i]:.4f} | y = {a[i]:.4f} * x + {b[i]:.4f}")


    #############################################