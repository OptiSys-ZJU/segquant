import os
from tqdm import tqdm


def trace_pic(model, path, data_loader, latents=None, max_num=None, **kwargs):
    os.makedirs(path, exist_ok=True)
    count = 0
    pbar = tqdm(
        total=max_num if max_num is not None else len(data_loader.dataset),
        desc="Tracing images",
    )
    for batch in data_loader:
        for b in batch:
            if max_num is not None and count >= max_num:
                pbar.close()
                return
            prompt, _, control = b
            image = model.forward(
                prompt=prompt, control_image=control, latents=latents, **kwargs
            )[0]
            save_path = os.path.join(path, f"{count:04d}.jpg")
            image[0].save(save_path)
            count += 1
            pbar.update(1)
    pbar.close()
