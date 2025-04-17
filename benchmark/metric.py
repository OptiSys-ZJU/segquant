import torch
import lpips
import numpy as np
from torchvision import transforms
from PIL import Image
from pytorch_fid import fid_score
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from pathlib import Path
import argparse

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


def run_bench(type):
    # scales = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    scales = ['0.2', '0.5', '0.8']
    
    total_fid = 0
    total_lpips = 0
    total_psnr = 0
    total_ssim = 0
    num_scales = len(scales)
    
    for scale in scales:
        print(scale)
        this_fid, group_lpips, group_psnr, group_ssim = calculate_all(f'pic/fp16/{scale}', f'pic/{type}/{scale}')
        
        total_fid += this_fid
        total_lpips += group_lpips
        total_psnr += group_psnr
        total_ssim += group_ssim
        
        print(f'type: {type}')
        print(f'fid: {this_fid:.4f}')
        print(f'group_lpips: {group_lpips:.4f}')
        print(f'group_psnr: {group_psnr:.4f}')
        print(f'group_ssim: {group_ssim:.4f}')
        print('============================================')
    
    avg_fid = total_fid / num_scales
    avg_lpips = total_lpips / num_scales
    avg_psnr = total_psnr / num_scales
    avg_ssim = total_ssim / num_scales

    print('Overall Results:')
    print(f'Average FID: {avg_fid:.4f}')
    print(f'Average LPIPS: {avg_lpips:.4f}')
    print(f'Average PSNR: {avg_psnr:.4f}')
    print(f'Average SSIM: {avg_ssim:.4f}')
    print('============================================')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Example of multiple arguments as a list")
    parser.add_argument('--types', nargs='+', help="List of Types", required=True)
    args = parser.parse_args()
    print("All types:", args.types)
    for type in args.types:
        run_bench(type)