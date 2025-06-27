import os
from tqdm import tqdm
import torch


def trace_pic(model, path=None, data_loader=None, latents=None, max_num=None, reprocess=False, **kwargs):
    os.makedirs(path, exist_ok=True)
    count = 0
    if max_num is not None:
        if max_num > len(data_loader.dataset):
            raise ValueError("max_num is greater than the number of images in the dataset")
    else:
        max_num = len(data_loader.dataset)
    
    if not reprocess:
        print(f"[INFO] reprocess is False, will continue processsing if there are images left undone")
        if len(os.listdir(path)) > max_num:
            raise ValueError("max_num is less than the number of images in the path")
        pbar = tqdm(
            total = max_num - len(os.listdir(path)),
            desc="Tracing images",
        )
    else:
        print(f"[INFO] reprocess is True, tracing all images")
        pbar = tqdm(
            total=max_num if max_num is not None else len(data_loader.dataset),
            desc="Tracing images",
        )
    
    # Get model device to ensure inputs are on the same device
    model_device = next(model.parameters()).device

    if not reprocess:
        continue_id = len(os.listdir(path))


    for batch in data_loader:
        for b in batch:
            if max_num is not None and count >= max_num:
                pbar.close()
                print("Max_num already reached, exiting...")
                return
            if not reprocess and count < continue_id:
                count += 1
                continue
            prompt, _, control = b
            # print(f"id: {id}, prompt: {prompt}") # testcode for debug
            # id += 1 # testcode for debug
            # Ensure control image is on the correct device
            if hasattr(control, 'to'):
                # It's already a tensor
                control = control.to(model_device)
            elif hasattr(control, 'device'):
                # It's a tensor but check if it needs moving
                if control.device != model_device:
                    control = control.to(model_device)
            
            # Set default tensor device for this operation
            with torch.cuda.device(model_device):
                image = model.forward(
                    prompt=prompt, control_image=control, latents=latents, **kwargs
                )[0]
            
            save_path = os.path.join(path, f"{count:04d}.jpg")
            image[0].save(save_path)
            count += 1
            pbar.update(1)
    pbar.close()
