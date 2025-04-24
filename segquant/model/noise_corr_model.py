import torch
import torch.nn as nn

from dataset.noise_diff.noise_diff_dataset import NoiseDiffDataset


def frobenius_loss(pred, target):
    diff = pred - target  # [B, C, H, W]
    loss = torch.norm(diff.view(diff.size(0), -1), dim=1)
    return loss.mean()

class NoiseCorrModel(nn.Module):
    def __init__(self, hidden_dim=256, window_size=3):
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
        pred_error = self.decoder(h_n).unsqueeze(1)  # (B, 1, 16, 128, 128)
        return pred_error

    def forward(self, history_noise_pred, history_timestep, history_controlnet_scale, history_guidance_scale):
        if isinstance(history_noise_pred, tuple):
            history_uncond, history_text = history_noise_pred
            history_timestep = history_timestep / 1000.0

            uncond_feats = self.__encode__(history_uncond)
            text_feats = self.__encode__(history_text)

            zero_guidance_scale = torch.zeros_like(history_guidance_scale)

            uncond_pred = self.__lstm__(uncond_feats, t, history_controlnet_scale, zero_guidance_scale)
            text_pred = self.__lstm__(text_feats, t, history_controlnet_scale, zero_guidance_scale)

            final_error = uncond_pred + history_guidance_scale[0][0].item() * (text_pred - uncond_pred)
        else:
            feats = self.__encode__(history_noise_pred)
            final_error = self.__lstm__(feats, t, history_controlnet_scale, history_guidance_scale)

        return final_error


if __name__ == '__main__':
    dataset = NoiseDiffDataset(data_dir='../noise_dataset')
    data_loader = dataset.get_dataloader(batch_size=2, shuffle=True)

    for batch in data_loader:
        print(type(batch['history_noise_pred']))
        print(batch['history_noise_pred'][0].shape)
        print(batch['history_noise_pred'][1].shape)
        print(batch['history_timestep'].shape)
        print(batch['history_timestep'])
        print(batch['history_controlnet_scale'].shape)
        print(batch['history_guidance_scale'].shape)
        print(batch['real_noise_pred'].shape)
        exit(0)