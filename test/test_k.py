import torch
import torch.nn.functional as F
from dataset.noise_diff.noise_diff_dataset import NoiseDiffDataset

def matrix_loss(K, epsilon_hat, epsilon, lambda1=0.5):
    """
    K:        shape [B, C, H, W], learnable scale
    epsilon_hat: predicted residual, shape [B, C, H, W]
    epsilon:     true residual, shape [B, C, H, W]
    """

    epsilon_tilde = K * epsilon_hat
    mse_loss = F.mse_loss(epsilon_tilde, epsilon, reduction='mean')
    numerator = torch.sum((epsilon_tilde - epsilon_hat) ** 2)
    denominator = torch.sum(epsilon ** 2) + 1e-8
    snr_like = numerator / denominator
    loss = (1 - lambda1) * mse_loss + lambda1 * snr_like

    return loss

def custom_loss(K, epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1):
    """
    K:        shape [B, C, H, W], learnable scale
    epsilon_hat: predicted residual, shape [B, C, H, W]
    epsilon:     true residual, shape [B, C, H, W]
    """

    epsilon_tilde = K * epsilon_hat
    mse_loss = F.mse_loss(epsilon_tilde, epsilon, reduction='mean')
    numerator = torch.sum((epsilon_tilde - epsilon_hat) ** 2)
    denominator = torch.sum(epsilon ** 2) + 1e-8
    snr_like = numerator / denominator
    k_penalty = torch.mean((K - 1.0) ** 2)
    loss = (1 - lambda1) * mse_loss + lambda1 * snr_like + lambda2 * k_penalty

    return loss

def solve_optimal_k(epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1):
    B, C, H, W = epsilon.shape
    N = C * H * W
    a = (1 - lambda1) * (epsilon_hat * epsilon).sum(dim=(2, 3)) + lambda1 * N * (epsilon_hat / epsilon).sum(dim=(2, 3)) + lambda2 * N
    b = (1 - lambda1) * (epsilon_hat ** 2).sum(dim=(2, 3)) + lambda1 * N * ((epsilon_hat ** 2) / (epsilon ** 2)).sum(dim=(2, 3)) + lambda2 * N

def solve_optimal_K_block(epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1, block=1):
    """
    epsilon_hat: [B, C, H, W]
    epsilon:     [B, C, H, W]
    return:      [B, C, H, W] K_opt with block-wise shared values
    """
    B, C, H, W = epsilon.shape
    D = torch.sum(epsilon ** 2) + 1e-8  # 全图整体标量

    A = (1 - lambda1) * epsilon_hat**2 + lambda1 * epsilon_hat**2 / D + lambda2
    B_term = -2 * (1 - lambda1) * epsilon_hat * epsilon - 2 * lambda1 * epsilon_hat**2 / D - 2 * lambda2
    K_pixelwise = -B_term / (2 * A + 1e-8)  # [B, C, H, W]

    if block == 1:
        return K_pixelwise  # 最细粒度，直接返回

    # 检查是否整除
    assert H % block == 0 and W % block == 0, "H and W must be divisible by block size"

    # reshape成 block 结构
    K_blocks = K_pixelwise.unfold(2, block, block).unfold(3, block, block)  # [B, C, H//b, W//b, b, b]
    K_blocks = K_blocks.contiguous().view(B, C, H // block, W // block, -1)  # flatten block
    K_block_mean = K_blocks.mean(dim=-1)  # [B, C, H//b, W//b]

    # broadcast 回原大小
    K_upsampled = F.interpolate(K_block_mean, size=(H, W), mode='nearest')  # [B, C, H, W]
    return K_upsampled

def matrix_loss_with_b(K, b, epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1):
    """
    Compute the custom loss including K and b, based on the original loss function.
    """
    # Calculate predicted epsilon
    epsilon_tilde = K * epsilon_hat + b
    
    # MSE loss
    mse_loss = torch.mean((epsilon_tilde - epsilon) ** 2)
    
    # SNR-like term (same as in the original function)
    numerator = torch.sum((epsilon_tilde - epsilon_hat) ** 2)
    denominator = torch.sum(epsilon ** 2) + 1e-8
    snr_like = numerator / denominator
    
    # Total loss
    loss = (1 - lambda1) * mse_loss + lambda1 * snr_like

    return loss

def custom_loss_with_b(K, b, epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1):
    """
    Compute the custom loss including K and b, based on the original loss function.
    """
    # Calculate predicted epsilon
    epsilon_tilde = K * epsilon_hat + b
    
    # MSE loss
    mse_loss = torch.mean((epsilon_tilde - epsilon) ** 2)
    
    # SNR-like term (same as in the original function)
    numerator = torch.sum((epsilon_tilde - epsilon_hat) ** 2)
    denominator = torch.sum(epsilon ** 2) + 1e-8
    snr_like = numerator / denominator
    
    # Regularization terms
    k_penalty = torch.mean((K - 1.0) ** 2)  # Encourage K to be close to 1
    b_penalty = torch.mean(b ** 2)  # Prevent b from becoming too large
    
    # Total loss
    loss = (1 - lambda1) * mse_loss + lambda1 * snr_like + lambda2 * k_penalty + lambda2 * b_penalty

    return loss

def solve_optimal_K_b(epsilon_hat, epsilon, lambda1=0.5, lambda2=0.1):
    """
    Solve for the optimal K and b given the input epsilon_hat and epsilon using closed-form solution.
    """
    # Initialize b to 0 initially
    b = torch.zeros_like(epsilon_hat)

    # Calculate K using the closed-form solution
    # Summing over B, H, W dimensions (dim=0, 2, 3)
    K_numerator = torch.sum(epsilon_hat * (epsilon - b), dim=(0, 2, 3))  # Summing over B, H, W
    K_denominator = torch.sum(epsilon_hat ** 2, dim=(0, 2, 3)) + lambda1  # Summing over B, H, W
    K = K_numerator / K_denominator

    # Ensure that K has the correct shape for broadcasting (should be [1, 16, 1, 1] for broadcasting)
    K = K.view(1, -1, 1, 1)

    # Calculate b using the closed-form solution
    b_numerator = torch.sum(epsilon - K * epsilon_hat, dim=(0, 2, 3))  # Summing over B, H, W
    b_denominator = epsilon_hat.shape[0] * epsilon_hat.shape[2] * epsilon_hat.shape[3] + lambda2  # B * H * W + lambda2
    b = b_numerator / b_denominator

    # Ensure b has the correct shape for broadcasting (should be [1, 16, 1, 1])
    b = b.view(1, -1, 1, 1)

    return K, b

def solve_optimal_K_b_block(epsilon_hat, epsilon, block=1, lambda1=0.5, lambda2=0.1):
    """
    Solve for block-wise optimal K and b.
    Args:
        epsilon_hat: [B, C, H, W]
        epsilon:     [B, C, H, W]
        block: block size along H and W
    Returns:
        K: [1, C, H, W] with block-wise shared values
        b: [1, C, H, W] with block-wise shared values
    """
    B, C, H, W = epsilon_hat.shape
    assert H % block == 0 and W % block == 0, "H and W must be divisible by block size"

    h_blocks = H // block
    w_blocks = W // block

    K_out = torch.zeros((1, C, H, W), device=epsilon_hat.device)
    b_out = torch.zeros((1, C, H, W), device=epsilon_hat.device)

    for i in range(h_blocks):
        for j in range(w_blocks):
            h_start = i * block
            h_end = (i + 1) * block
            w_start = j * block
            w_end = (j + 1) * block

            # Slice block
            e_hat_block = epsilon_hat[:, :, h_start:h_end, w_start:w_end]
            e_block = epsilon[:, :, h_start:h_end, w_start:w_end]

            # 1. Solve K
            K_numerator = torch.sum(e_hat_block * e_block, dim=(0, 2, 3))
            K_denominator = torch.sum(e_hat_block ** 2, dim=(0, 2, 3)) + lambda1
            K = K_numerator / (K_denominator + 1e-8)  # shape [C]
            K = K.view(1, C, 1, 1)

            # 2. Solve b
            b_numerator = torch.sum(e_block - K * e_hat_block, dim=(0, 2, 3))
            b_denominator = e_hat_block.shape[0] * e_hat_block.shape[2] * e_hat_block.shape[3] + lambda2
            b = b_numerator / (b_denominator + 1e-8)
            b = b.view(1, C, 1, 1)

            # 3. Fill block
            K_out[:, :, h_start:h_end, w_start:w_end] = K
            b_out[:, :, h_start:h_end, w_start:w_end] = b

    return K_out, b_out

def to_device(obj, device):
    if isinstance(obj, tuple):
        return tuple(x.to(device) for x in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

if __name__ == '__main__':
    device = 'cuda'
    path = 'noise_dataset_dit'
    dataset = NoiseDiffDataset(data_dir=f'../{path}/train')
    print(f"Dataset loaded: {len(dataset)} samples")

    step = 0

    ks = {}
    ks_b = {}
    block_sizes = [1,2,4,8,16,32,64,128]
    for block in block_sizes:
        ks[block] = {}
        ks_b[block] = {}

    sample = 32
    max_steps = 57 * sample
    print('Max steps (samples):', max_steps)

    for batch in dataset.get_dataloader(batch_size=1, shuffle=False):
        history_timestep = to_device(batch['history_timestep'], device)
        timestep = history_timestep[-1, -1, 0].item()
        real = to_device(batch['real_noise_pred'].squeeze().unsqueeze(0), device)
        quant = to_device(batch['quant_noise_pred'].squeeze().unsqueeze(0), device)

        print(f"\nStep {step + 1}, Timestep: {timestep}")

        for block in block_sizes:
            K_opt = solve_optimal_K_block(quant, real, block=block)
            loss_opt = custom_loss(K_opt, quant, real)
            print(f"  [block={block}] K-only Loss: {loss_opt.item():.6f}")

            if f'{timestep}' not in ks[block]:
                ks[block][f'{timestep}'] = K_opt
            else:
                ks[block][f'{timestep}'] += K_opt

            K_opt_k, K_opt_b = solve_optimal_K_b_block(quant, real, block=block)
            loss_opt_b = custom_loss_with_b(K_opt_k, K_opt_b, quant, real)
            print(f"  [block={block}] K-b Loss: {loss_opt_b.item():.6f}")

            if f'{timestep}' not in ks_b[block]:
                ks_b[block][f'{timestep}'] = (K_opt_k, K_opt_b)
            else:
                ks_b[block][f'{timestep}'] = (
                    ks_b[block][f'{timestep}'][0] + K_opt_k,
                    ks_b[block][f'{timestep}'][1] + K_opt_b
                )

        step += 1
        if step >= max_steps:
            break

    print("\nAveraging results...")
    for block in block_sizes:
        for k, v in ks[block].items():
            ks[block][k] = v / sample
        for k, v in ks_b[block].items():
            ks_b[block][k] = (v[0] / sample, v[1] / sample)

    print("Finished. All Ks and Bs are averaged.")
    ks_mean = ks
    ks_b_mean = ks_b

    step = 0
    max_steps = 100
    for batch in dataset.get_dataloader(batch_size=1, shuffle=True):
        history_timestep = to_device(batch['history_timestep'], device)
        timestep = history_timestep[-1, -1, 0]
        real = to_device(batch['real_noise_pred'].squeeze().unsqueeze(0), device)
        quant = to_device(batch['quant_noise_pred'].squeeze().unsqueeze(0), device)

        K = torch.ones_like(real)
        loss1 = custom_loss(K, quant, real)
        loss_1_matrix = matrix_loss(K, quant, real)

        K = torch.ones_like(real)
        b = torch.zeros_like(real)
        loss1_b = custom_loss_with_b(K, b, quant, real)
        loss_1_matrix_b = matrix_loss_with_b(K, b, quant, real)

        for block in block_sizes:
            K = ks_mean[block][f'{timestep}']
            loss2 = custom_loss(K, quant, real)
            loss2_matrix = matrix_loss(K, quant, real)

            K_opt = solve_optimal_K_block(quant, real, block=block)
            loss_opt = custom_loss(K_opt, quant, real)
            loss_opt_matrix = matrix_loss(K_opt, quant, real)

            print(f"[{step}]-['{timestep}']-[{block}] K-only Init[{loss1.item():.6f}/{loss_1_matrix.item():.6f}], Affine[{loss2.item():.6f}/{loss2_matrix.item():.6f}], OPT[{loss_opt.item():.6f}/{loss_opt_matrix.item():.6f}]")

            ####################################################################
            K, b = ks_b_mean[block][f'{timestep}']
            loss2_b = custom_loss_with_b(K, b, quant, real)
            loss2_matrix_b = matrix_loss_with_b(K, b, quant, real)
            
            K_opt_k, K_opt_b = solve_optimal_K_b_block(quant, real, block=block)
            loss_opt_b = custom_loss_with_b(K_opt_k, K_opt_b, quant, real)
            loss_opt_matrix_b = matrix_loss_with_b(K_opt_k, K_opt_b, quant, real)
            
            print(f"[{step}]-['{timestep}']-[{block}] K-b Init[{loss1_b.item():.6f}/{loss_1_matrix_b.item():.6f}], Affine[{loss2_b.item():.6f}/{loss2_matrix_b.item():.6f}], OPT[{loss_opt_b.item():.6f}/{loss_opt_matrix_b.item():.6f}]")

        print('-----------------------------------')
        step += 1
        if step >= max_steps:
            break