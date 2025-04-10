import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import re
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.nn.functional import mse_loss
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import matplotlib.pyplot as plt

def frobenius(A, B):
    return torch.norm(A - B, p='fro')

def frobenius_loss(pred, target):
    diff = pred - target  # [B, C, H, W]
    loss = torch.norm(diff.view(diff.size(0), -1), dim=1)  # 每个样本一个 Frobenius
    return loss.mean()  # batch 平均

class QuantErrorPredictor(nn.Module):
    def __init__(self, hidden_dim=256, window_size=3):
        super().__init__()
        self.window_size = window_size

        # 编码器：降维但保留空间信息
        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(),
        )
        self.flatten_dim = 64 * 32 * 32

        # LSTM 接收时间序列的编码
        self.lstm = nn.LSTM(input_size=self.flatten_dim + 2, hidden_size=hidden_dim, batch_first=True)

        # 输出层：映射回误差图像
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 32 * 32),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 128, 128)
            nn.Tanh(),
        )
    
    def __encode__(self, noise):
        B, T, C, H, W = noise.shape  # B=batch, T=window_size
        x = noise.view(B * T, C, H, W)  # 合并时间维度，逐帧送入 CNN
        feats = self.encoder(x)  # (B*T, 64, 32, 32)
        feats = feats.view(B, T, -1)  # (B, T, flatten_dim)
        return feats

    def __lstm__(self, feat, t, cond_scale):
        lstm_input = torch.cat([feat, t, cond_scale], dim=-1)  # (B, T, flatten_dim + 1)
        _, (h_n, _) = self.lstm(lstm_input)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)
        pred_error = self.decoder(h_n).unsqueeze(1)  # (B, 1, 16, 128, 128)
        return pred_error

    def forward(self, history_quant, history_timestep, cond_scale, guidance_scale):
        history_timestep = history_timestep / 1000.0
        history_uncond, history_text = history_quant

        t = history_timestep.unsqueeze(-1)  # (B, T, 1)

        uncond_feats = self.__encode__(history_uncond)
        text_feats = self.__encode__(history_text)
        
        uncond_pred = self.__lstm__(uncond_feats, t, cond_scale)
        text_pred = self.__lstm__(text_feats, t, cond_scale)

        final_error = uncond_pred + guidance_scale[0][0].item() * (text_pred - uncond_pred)

        return final_error


class QuantErrorSequenceDataset(Dataset):
    def __init__(self, dir, window_size=3, type='int8_smooth_enablelatent', device="cpu"):
        """
        Dataset for feeding LSTM-like models with time-windowed quantization error sequences.
        
        :param dir: root directory containing the fp16 and int8 outputs.
        :param window_size: number of past steps to use as input.
        :param type: quantized result prefix (e.g., 'int8_smooth_enablelatent').
        :param device: torch device.
        """
        self.dir = dir
        self.device = device
        self.type = type
        self.window_size = window_size
        self.sequence_data = []

        # Step 1: Load and sort data by timestep
        raw_data = {}
        regex = re.compile('fp16.*\.pt')
        for root, dirs, files in os.walk(self.dir):
            for file in files:
                if regex.match(file):
                    fp16_path = os.path.join(root, file)
                    quant_path = os.path.join(root, file.replace('fp16', type))
                    if not os.path.exists(quant_path):
                        print(f"Skip [{fp16_path}]")
                        continue
                    fp16 = torch.load(fp16_path)
                    quant = torch.load(quant_path)

                    guid_scale = str(fp16['guidance_scale'])
                    cond_scale = str(fp16['cond_scale'])
                    if guid_scale not in raw_data:
                        raw_data[guid_scale] = {}
                    if cond_scale not in raw_data[guid_scale]:
                        raw_data[guid_scale][cond_scale] = []

                    raw_data[guid_scale][cond_scale].append({
                        'fp_output': fp16['noise_pred'][0],

                        'quant_uncond_output': quant['noise_pred_uncond'][0],
                        'quant_text_output': quant['noise_pred_text'][0],
                        'quant_output': quant['noise_pred'][0],

                        'timestep': float(fp16['timestep'][0]),
                        'cond_scale': float(fp16['cond_scale']),
                        'guidance_scale': float(fp16['guidance_scale']),
                    })

        for d in raw_data.values():
            for l in d.values():
                l.sort(key=lambda x: x['timestep'], reverse=True)

                for i in range(window_size, len(l)):
                    history = l[i - window_size + 1 : i + 1]
                    target = l[i]
                    self.sequence_data.append({
                        'cond_scale': [h['cond_scale'] for h in history],
                        'guid_scale': [h['guidance_scale'] for h in history],

                        'history_quant': ([h['quant_uncond_output'] for h in history], [h['quant_text_output'] for h in history]),
                        'history_timestep': [h['timestep'] for h in history],

                        'target_error': target['fp_output'] - target['quant_output'],
                        'target_timestep': target['timestep'],
                        'target_output': target['quant_output'],
                        'fp_output': target['fp_output'],
                    })

    def __len__(self):
        return len(self.sequence_data)

    def __getitem__(self, idx):
        sample = self.sequence_data[idx]

        # Convert list of tensors to single tensor: shape (window_size, 16, 128, 128)
        a, b = sample['history_quant']
        history_quant_tensor = (torch.stack(a, dim=0), torch.stack(b, dim=0))
        history_timestep_tensor = torch.tensor(sample['history_timestep'], dtype=torch.float32)
        history_cond_tensor = torch.tensor(sample['cond_scale'], dtype=torch.float32)
        history_guidance_tensor = torch.tensor(sample['guid_scale'], dtype=torch.float32)

        return {
            'history_quant': history_quant_tensor,                      # (window, 16, 128, 128)
            'history_timestep': history_timestep_tensor,                # (window,)
            'history_cond': history_cond_tensor.unsqueeze(-1),
            'history_guidance': history_guidance_tensor, 
            'target_error': sample['target_error'].unsqueeze(0),        # (1, 16, 128, 128)
            'target_output': sample['target_output'].unsqueeze(0),
            'target_timestep': sample['target_timestep'],
            'fp_output': sample['fp_output'].unsqueeze(0),
        }
    


def evaluate_model_fp16(model, dataloader, device='cuda'):
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    loss_fn = frobenius_loss

    with torch.no_grad():  # 评估时不计算梯度
        for batch in dataloader:
            history_quant = (batch['history_quant'][0].to(device), batch['history_quant'][1].to(device))
            history_timestep = batch['history_timestep'].to(device)
            history_cond = batch['history_cond'].to(device)
            history_guidance = batch['history_guidance'].to(device)
            target_error = batch['target_error'].to(device)

            # 混合精度前向传播
            with autocast(device):
                pred_error = model(history_quant, history_timestep, history_cond, history_guidance)     # (B, 1, 16, 128, 128)
                loss = loss_fn(pred_error, target_error)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Evaluation Loss: {avg_loss:.6f}")
    model.train()  # 训练后恢复模型为训练模式
    return avg_loss

def save_model(model, optimizer, epoch, filepath):
    """
    保存模型和优化器的状态字典。
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Model saved to {filepath}")

@torch.no_grad()
def run_inference(model_path, data_dir, device='cuda'):
    # 加载模型
    model = QuantErrorPredictor(hidden_dim=256, window_size=3)
    model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 加载数据
    dataset = QuantErrorSequenceDataset(
        dir=data_dir,
        window_size=3,
        type='int8_smooth_enablelatent',
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    total_mse = 0.0
    count = 0

    for i, batch in enumerate(tqdm(dataloader, desc="Running inference")):
        history_quant = (batch['history_quant'][0].to(device), batch['history_quant'][1].to(device))
        history_timestep = batch['history_timestep'].to(device)
        history_cond = batch['history_cond'].to(device)
        history_guidance = batch['history_guidance'].to(device)
        target_error = batch['target_error'].to(device)
        target_output = batch['target_output'].to(device)
        target_timestep = batch['target_timestep']

        with autocast(device_type='cuda'):
            pred_error = model(history_quant, history_timestep, history_cond, history_guidance)   # (B, 1, 16, 128, 128)
            corrected_output = target_output + pred_error.squeeze(1)

        if 'target_output' in batch:
            fp_output = batch['fp_output'].to(device)

            res1 = frobenius(fp_output, target_output)
            res2 = frobenius(fp_output, corrected_output)
            print(f"[Sample {i}] timestep={target_timestep[0].item():.3f} | Prev={res1:.4f} | After={res2:.4f}")


def overfit_on_sample(model, dataset, device='cuda', epochs=1000, lr=1e-4, use_amp=True):
    """
    尝试在一个样本上进行过拟合，验证模型是否有足够的表达能力。
    
    :param model: 要训练的模型
    :param dataset: QuantErrorSequenceDataset 实例
    :param device: 训练设备（如 'cuda'）
    :param epochs: 训练轮数
    :param lr: 学习率
    :param use_amp: 是否使用混合精度
    """
    model = model.to(device)
    model.train()

    # 只取第一个样本，构造 DataLoader
    sample = dataset[0]
    dataloader = DataLoader([sample], batch_size=1, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = GradScaler(enabled=use_amp)
    loss_fn = torch.nn.MSELoss()

    print("🔍 开始 Overfit 实验...")
    for epoch in range(epochs):
        for batch in dataloader:
            history_quant = (batch['history_quant'][0].to(device), batch['history_quant'][1].to(device))
            history_timestep = batch['history_timestep'].to(device)
            history_cond = batch['history_cond'].to(device)
            history_guidance = batch['history_guidance'].to(device)
            target_error = batch['target_error'].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device, enabled=use_amp):
                pred_error = model(history_quant, history_timestep, history_cond, history_guidance)
                loss = loss_fn(pred_error, target_error)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.6f}")

    print("✅ Overfit 实验结束！")

def overfit_on_samples(model, dataset, sample_indices=[0, 1, 2], epochs=1000, lr=1e-3, device='cuda'):
    """
    Overfit 模型在指定的几个样本上，验证模型是否能记住它们。
    """
    

    # 构建子数据集
    subset = Subset(dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=1, shuffle=False)

    model = model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    scaler = GradScaler()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            history_quant = (batch['history_quant'][0].to(device), batch['history_quant'][1].to(device))
            history_timestep = batch['history_timestep'].to(device)
            history_cond = batch['history_cond'].to(device)
            history_guidance = batch['history_guidance'].to(device)
            target_error = batch['target_error'].to(device)

            optimizer.zero_grad()
            with autocast(device_type=device):
                pred_error = model(history_quant, history_timestep, history_cond, history_guidance)
                loss = loss_fn(pred_error, target_error)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.6f}")

def train_model_fp16_with_plot(model, dataset, epochs=100, lr=1e-4, batch_size=1, device='cuda', eval_every=1, save_path="model.pth"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = frobenius_loss
    scaler = GradScaler()

    loss_history = []
    best_loss = 1

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"[Epoch {epoch+1}/{epochs}]")
        for batch in pbar:
            history_quant = (batch['history_quant'][0].to(device), batch['history_quant'][1].to(device))
            history_timestep = batch['history_timestep'].to(device)
            history_cond = batch['history_cond'].to(device)
            history_guidance = batch['history_guidance'].to(device)
            target_error = batch['target_error'].to(device)

            optimizer.zero_grad()
            with autocast(device):
                pred_error = model(history_quant, history_timestep, history_cond, history_guidance)
                loss = loss_fn(pred_error, target_error)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")

        # 每隔 eval_every 轮进行评估
        if (epoch + 1) % eval_every == 0:
            print(f"Evaluating model at epoch {epoch+1}")
            eval_loss = evaluate_model_fp16(model, dataloader, device)

            # 如果当前模型的评估损失更低，则保存模型
            if eval_loss < best_loss:
                best_loss = eval_loss
                save_model(model, optimizer, epoch, save_path)

    # 保存模型
    torch.save(model.state_dict(), "model.pth")

    # 绘制 loss 曲线
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig("loss_curve.png")

if __name__ == '__main__':

    dataset = QuantErrorSequenceDataset(
        dir='train_lstm',
        window_size=3,
        type='int8_smooth_enablelatent',
    )
    model = QuantErrorPredictor(hidden_dim=256, window_size=3)

    # overfit_on_sample(model, dataset, epochs=10000)
    # overfit_on_samples(model, dataset, sample_indices=[0, 10, 20], epochs=10000, lr=1e-3, device='cuda')
    train_model_fp16_with_plot(model, dataset, epochs=200, lr=1e-5, batch_size=1, device='cuda', eval_every=1)

    # run_inference(
    #     model_path='model.pth',
    #     data_dir='train_lstm',
    #     device='cuda'
    # )
