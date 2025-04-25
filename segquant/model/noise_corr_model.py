from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataset.noise_diff.noise_diff_dataset import NoiseDiffDataset


def frobenius_loss(pred, target):
    diff = pred - target  # [B, C, H, W]
    loss = torch.norm(diff.view(diff.size(0), -1), dim=1)
    return loss.mean()

class NoiseCorrModel(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32)
            nn.ReLU(),
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

            final_error = uncond_pred + history_guidance_scale[:, 0].view(-1, 1, 1, 1) * (text_pred - uncond_pred)
        else:
            feats = self.__encode__(history_noise_pred)
            final_error = self.__lstm__(feats, history_timestep, history_controlnet_scale, history_guidance_scale)

        return final_error

def to_device(obj, device):
    if isinstance(obj, tuple):
        return tuple(x.to(device) for x in obj)
    elif isinstance(obj, torch.Tensor):
        return obj.to(device)
    else:
        raise TypeError(f"Unsupported type: {type(obj)}")

def evaluate(model, batch_size, device='cuda'):
    dataset = NoiseDiffDataset(data_dir='../noise_dataset/val')
    data_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)

    model.eval()
    total_loss = 0.0
    loss_fn = frobenius_loss

    with torch.no_grad():
        for batch in data_loader:
            real = to_device(batch['real_noise_pred'].squeeze(), device)
            quant = to_device(batch['quant_noise_pred'].squeeze(), device)
            target = real - quant

            history_noise_pred = to_device(batch['history_noise_pred'], device)
            history_timestep = to_device(batch['history_timestep'], device)
            history_controlnet_scale = to_device(batch['history_controlnet_scale'], device)
            history_guidance_scale = to_device(batch['history_guidance_scale'], device)

            res = model.forward(history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale)
            
            loss = loss_fn(target, res)

            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Evaluation Loss: {avg_loss:.6f}")
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

    model = NoiseCorrModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
        cooldown=3, min_lr=1e-6
    )

    loss_fn = frobenius_loss

    loss_history = []
    best_loss = None

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        pbar = tqdm(data_loader, desc=f"[Epoch {epoch+1}/{epochs}]")
        for batch in pbar:
            real = to_device(batch['real_noise_pred'].squeeze(), device)
            quant = to_device(batch['quant_noise_pred'].squeeze(), device)
            target = real - quant

            history_noise_pred = to_device(batch['history_noise_pred'], device)
            history_timestep = to_device(batch['history_timestep'], device)
            history_controlnet_scale = to_device(batch['history_controlnet_scale'], device)
            history_guidance_scale = to_device(batch['history_guidance_scale'], device)

            res = model.forward(history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale)
            
            loss = loss_fn(target, res)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item(), lr=scheduler.get_last_lr()[0])
        
        avg_loss = total_loss / len(data_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch+1}] Avg Loss: {avg_loss:.6f}")
        if (epoch + 1) % eval_every == 0:
            print(f"Evaluating model at epoch {epoch+1}")
            eval_loss = evaluate(model, batch_size, device)

            scheduler.step(eval_loss)

            if best_loss is None:
                best_loss = eval_loss
                save_model(model, optimizer, epoch, save_path)
            else:
                if eval_loss < best_loss:
                    best_loss = eval_loss
                    save_model(model, optimizer, epoch, save_path)
    
    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig("loss_curve.png")


if __name__ == '__main__':
    train(epochs=1000, lr=5e-4, batch_size=32, device='cuda', eval_every=1)