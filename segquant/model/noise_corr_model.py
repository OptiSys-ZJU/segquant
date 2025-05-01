from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import get_cosine_schedule_with_warmup

from dataset.noise_diff.noise_diff_dataset import NoiseDiffDataset


def frobenius_loss(pred, target):
    diff = pred - target  # [B, C, H, W]
    loss = torch.norm(diff.view(diff.size(0), -1), dim=1)
    return loss.mean()

class MSE_QNSR_Loss(nn.Module):
    def __init__(self, lambda_mse=0.5, lambda_qnsr=0.5, epsilon=1e-8):
        super(MSE_QNSR_Loss, self).__init__()
        self.lambda_mse = lambda_mse
        self.lambda_qnsr = lambda_qnsr
        self.epsilon = epsilon
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        """
        pred: (batch, channel, h, w)
        target: (batch, channel, h, w)
        """
        mse = self.mse_loss(pred, target)

        numerator = ((pred - target) ** 2).sum()
        denominator = (target ** 2).sum() + self.epsilon
        qnsr = numerator / denominator

        loss = self.lambda_mse * mse + self.lambda_qnsr * qnsr
        return loss

class NoiseCorrModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(),
            # nn.Dropout(dropout_rate)
        )
        self.flatten_dim = 64 * 32 * 32

        self.lstm = nn.LSTM(input_size=self.flatten_dim + 3, hidden_size=hidden_dim, batch_first=True)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64 * 32 * 32),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # (16, 128, 128)
            nn.Tanh(),
            # nn.Dropout(dropout_rate)
        )
    
    def __encode__(self, noise):
        B, T, C, H, W = noise.shape  # B=batch, T=window_size
        x = noise.view(B * T, C, H, W)
        feats = self.encoder(x)  # (B*T, 64, 32, 32)
        feats = feats.view(B, T, -1)  # (B, T, flatten_dim)
        return feats

    def __lstm__(self, feat, t, cond_scale, guidance_scale):
        lstm_input = torch.cat([feat, t, cond_scale, guidance_scale], dim=-1)  # (B, T, flatten_dim + 1)
        _, (h_n, _) = self.lstm(lstm_input)
        h_n = h_n.squeeze(0)  # (B, hidden_dim)
        pred_error = self.decoder(h_n)  # (B, 16, 128, 128)
        return pred_error

    def forward(self, history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale):
        if isinstance(history_noise_pred, tuple):
            history_uncond, history_text = history_noise_pred
            history_timestep = history_timestep / 1000.0

            uncond_feats = self.__encode__(history_uncond)
            text_feats = self.__encode__(history_text)

            zero_guidance_scale = torch.zeros_like(history_guidance_scale)

            uncond_pred = self.__lstm__(uncond_feats, history_timestep, history_controlnet_scale, zero_guidance_scale)
            text_pred = self.__lstm__(text_feats, history_timestep, history_controlnet_scale, zero_guidance_scale)

            uncond_noise = uncond_pred * history_uncond[:, -1, :, :, :]
            text_noise = text_pred * history_text[:, -1, :, :, :]

            final_noise = uncond_noise + history_guidance_scale[:, 0].view(-1, 1, 1, 1) * (uncond_noise - text_noise)
        else:
            feats = self.__encode__(history_noise_pred)
            final_error = self.__lstm__(feats, history_timestep, history_controlnet_scale, history_guidance_scale)
            final_noise = final_error * history_noise_pred[:, -1, :, :, :]

        return final_noise
class NoiseCorrTransformerModel(nn.Module):
    def __init__(self, hidden_dim=256, max_seq_len=10):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU()
        )
        self.flatten_dim = 64 * 32 * 32  # =65536
        self.cond_dim = 3  # timestep, controlnet_scale, guidance_scale
        self.input_dim = hidden_dim  # Transformer输入维度
        self.input_linear = nn.Linear(self.flatten_dim + self.cond_dim, self.input_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, self.input_dim))  # (1, T, D)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = TransformerEncoderLayer(
            d_model=self.input_dim,
            nhead=8,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=3)
        self.decoder = nn.Sequential(
            nn.Linear(self.input_dim, 64 * 32 * 32),
            nn.Unflatten(1, (64, 32, 32)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def __encode__(self, noise):
        B, T, C, H, W = noise.shape
        x = noise.view(B * T, C, H, W)
        feats = self.encoder(x)  # (B*T, 64, 32, 32)
        feats = feats.view(B, T, -1)  # (B, T, flatten_dim)
        return feats

    def __transformer__(self, feat, t, cond_scale, guidance_scale):
        B, T, _ = feat.shape
        cond = torch.cat([t, cond_scale, guidance_scale], dim=-1)  # (B, T, 3)
        x = torch.cat([feat, cond], dim=-1)  # (B, T, flatten_dim + 3)
        x = self.input_linear(x)  # (B, T, input_dim)
        x = x + self.pos_embed[:, :T, :]  # 加上位置编码
        x = self.transformer(x)  # (B, T, input_dim)
        x_last = x[:, -1, :]  # (B, input_dim)
        pred_error = self.decoder(x_last)  # (B, 16, 128, 128)
        return pred_error

    def forward(self, history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale):
        history_timestep = history_timestep / 1000.0  # 归一化

        if isinstance(history_noise_pred, tuple):
            history_uncond, history_text = history_noise_pred

            uncond_feats = self.__encode__(history_uncond)
            text_feats = self.__encode__(history_text)

            zero_guidance_scale = torch.zeros_like(history_guidance_scale)

            uncond_pred = self.__transformer__(uncond_feats, history_timestep, history_controlnet_scale, zero_guidance_scale)
            text_pred = self.__transformer__(text_feats, history_timestep, history_controlnet_scale, zero_guidance_scale)

            uncond_noise = uncond_pred * history_uncond[:, -1, :, :, :]
            text_noise = text_pred * history_text[:, -1, :, :, :]

            final_noise = uncond_noise + history_guidance_scale[:, 0].view(-1, 1, 1, 1) * (uncond_noise - text_noise)
        else:
            feats = self.__encode__(history_noise_pred)
            final_error = self.__transformer__(feats, history_timestep, history_controlnet_scale, history_guidance_scale)
            final_noise = final_error * history_noise_pred[:, -1, :, :, :]

        return final_noise

def to_device(obj, device):
    if isinstance(obj, tuple):
        return tuple(x.to(device) for x in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

def evaluate_init(batch_size, path, device='cuda'):
    dataset = NoiseDiffDataset(data_dir=f'../noise_dataset/{path}')
    data_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    loss_fn = MSE_QNSR_Loss(lambda_mse=0.5, lambda_qnsr=0.5)

    with torch.no_grad():
        for batch in data_loader:
            real = to_device(batch['real_noise_pred'].squeeze(), device)
            quant = to_device(batch['quant_noise_pred'].squeeze(), device)
            target = real
            
            loss = loss_fn(target, quant)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Init {path} Loss: {avg_loss:.6f}")
    return avg_loss

def evaluate(model, batch_size, path, device='cuda'):
    dataset = NoiseDiffDataset(data_dir=f'../noise_dataset/{path}')
    data_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    model.eval()
    total_loss = 0.0
    loss_fn = MSE_QNSR_Loss(lambda_mse=0.5, lambda_qnsr=0.5)

    with torch.no_grad():
        for batch in data_loader:
            real = to_device(batch['real_noise_pred'].squeeze(), device)
            quant = to_device(batch['quant_noise_pred'].squeeze(), device)
            target = real

            history_noise_pred = to_device(batch['history_noise_pred'], device)
            history_timestep = to_device(batch['history_timestep'], device)
            history_controlnet_scale = to_device(batch['history_controlnet_scale'], device)
            history_guidance_scale = to_device(batch['history_guidance_scale'], device)

            res = model.forward(history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale)
            
            loss = loss_fn(target, res)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation {path} Loss: {avg_loss:.6f}")
    model.train()
    return avg_loss

def save_model(model, optimizer, epoch, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)
    print(f"Model saved to {filepath}")

def train(epochs=100, lr=1e-4, batch_size=1, device='cuda', eval_every=1, save_path="noise_corr_model.pth"):
    dataset = NoiseDiffDataset(data_dir='../noise_dataset/train')
    data_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    # model = NoiseCorrModel(hidden_dim=256, dropout_rate=0.1).to(device)
    model = NoiseCorrTransformerModel(hidden_dim=256).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    total_steps = epochs * len(data_loader)
    num_warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps
    )

    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=5,
    #     cooldown=3, min_lr=1e-5
    # )

    evaluate_init(batch_size, 'val', device)
    evaluate_init(batch_size, 'train', device)

    loss_fn = MSE_QNSR_Loss(lambda_mse=0.5, lambda_qnsr=0.5)

    loss_history = []
    best_loss = float('inf')
    global_step = 0

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"[Epoch {epoch+1}/{epochs}]")

        for batch in pbar:
            real = to_device(batch['real_noise_pred'].squeeze(), device)
            quant = to_device(batch['quant_noise_pred'].squeeze(), device)
            target = real

            history_noise_pred = to_device(batch['history_noise_pred'], device)
            history_timestep = to_device(batch['history_timestep'], device)
            history_controlnet_scale = to_device(batch['history_controlnet_scale'], device)
            history_guidance_scale = to_device(batch['history_guidance_scale'], device)

            pred = model(history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            global_step += 1

            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])

        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Train Loss: {avg_loss:.6f}")

        if (epoch + 1) % eval_every == 0:
            print(f"Evaluating at epoch {epoch+1}")
            val_loss = evaluate(model, batch_size, 'val', device)
            print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                save_model(model, optimizer, epoch + 1, save_path)
                print(f"✅ New best model saved with val loss: {best_loss:.6f}")

    plt.figure()
    plt.plot(loss_history, label="Train Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.legend()
    plt.savefig("loss_curve.png")

if __name__ == '__main__':
    train(epochs=1000, lr=1e-4, batch_size=32, device='cuda', eval_every=5)