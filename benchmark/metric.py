import torch
import lpips
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path

lpips_model = lpips.LPIPS(net='alex')

def calculate_fid(real_dir, fake_dir, batch_size=50, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    fid_value = fid_score.calculate_fid_given_paths([real_dir, fake_dir], 
                                                    batch_size=batch_size, 
                                                    device=device, 
                                                    dims=2048)
    return fid_value

def calculate_lpips(img1, img2):
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    img1 = transform(img1).unsqueeze(0) * 2 - 1
    img2 = transform(img2).unsqueeze(0) * 2 - 1
    
    lpips_score = lpips_model(img1, img2)
    return lpips_score.item()

def calculate_psnr(img1, img2):
    transform = transforms.ToTensor()
    img1 = transform(img1).numpy()
    img2 = transform(img2).numpy()
    
    return psnr(img1, img2, data_range=img1.max() - img1.min())


def calculate_ssim(img1, img2):
    img1 = np.array(img1.convert('L'))
    img2 = np.array(img2.convert('L'))
    
    return ssim(img1, img2, data_range=img1.max() - img1.min())

def calculate_all(real_path, this_path):
    this_fid = calculate_fid(real_path, this_path)
    group_lpips = []
    group_psnr = []
    group_ssim = []

    real_list = {}
    this_list = {}
    for f in Path(real_path).iterdir():
        if f.is_file():
            img = Image.open(f)
            step = f.name.split('-')[2]
            real_list[step] = img
    
    for f in Path(this_path).iterdir():
        if f.is_file():
            img = Image.open(f)
            step = f.name.split('-')[2]
            this_list[step] = img
    
    for key, real_img in real_list.items():
        this_img = this_list[key]
        this_lpips = calculate_lpips(real_img, this_img)
        this_psnr = calculate_psnr(real_img, this_img)
        this_ssim = calculate_ssim(real_img, this_img)
        group_lpips.append(this_lpips)
        group_psnr.append(this_psnr)
        group_ssim.append(this_ssim)
    

    return this_fid, sum(group_lpips) / len(group_lpips), sum(group_psnr) / len(group_psnr), sum(group_ssim) / len(group_ssim)

if __name__ == '__main__':
    this_fid, group_lpips, group_psnr, group_ssim = calculate_all('pic/fp16/0.8', 'pic/int8_default/0.8')
    print('int8_default')
    print('fid:', this_fid)
    print('group_lpips:', group_lpips)
    print('group_psnr:', group_psnr)
    print('group_ssim:', group_ssim)
    print("---------------------")
    this_fid, group_lpips, group_psnr, group_ssim = calculate_all('pic/fp16/0.8', 'pic/int8_smoothquant/0.8')
    print('int8_smoothquant')
    print('fid:', this_fid)
    print('group_lpips:', group_lpips)
    print('group_psnr:', group_psnr)
    print('group_ssim:', group_ssim)