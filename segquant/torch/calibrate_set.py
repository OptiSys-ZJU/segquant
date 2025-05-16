import io
import os
import zstandard as zstd
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class BaseCalibSet(Dataset):
    def __init__(self, data=None, folder=None, compress=False, max_cache_size=1):
        super().__init__()
        self.compress = compress
        self.max_cache_size = max_cache_size

        if data is not None:
            self.data = data
            self.folder = None
            self.chunk_files = None
        elif folder is not None:
            self.data = None
            self.folder = folder
            suffix = '.pt.zst' if self.compress else '.pt'
            self.chunk_files = sorted([
                os.path.join(folder, f) for f in os.listdir(folder)
                if f.endswith(suffix)
            ])
            self.chunk_lens = []
            for f in self.chunk_files:
                d = self._load_chunk(f)
                self.chunk_lens.append(len(d))
                del d
            self.cache = OrderedDict()  # chunk_idx -> chunk_data
        else:
            raise ValueError("Either data or folder must be provided.")

    def _load_chunk(self, path):
        if not self.compress:
            return torch.load(path)
        else:
            with open(path, 'rb') as f:
                compressed_bytes = f.read()
            dctx = zstd.ZstdDecompressor()
            decompressed_bytes = dctx.decompress(compressed_bytes)
            return torch.load(io.BytesIO(decompressed_bytes))

    def __len__(self):
        if self.data is not None:
            return len(self.data)
        else:
            return sum(self.chunk_lens)

    def __getitem__(self, idx):
        if self.data is not None:
            return self.data[idx]

        for chunk_idx, length in enumerate(self.chunk_lens):
            if idx < length:
                if chunk_idx not in self.cache:
                    if len(self.cache) >= self.max_cache_size:
                        self.cache.popitem(last=False)
                    self.cache[chunk_idx] = self._load_chunk(self.chunk_files[chunk_idx])
                return self.cache[chunk_idx][idx]
            else:
                idx -= length

        raise IndexError("Index out of range")

    @staticmethod
    def collate_fn(batch):
        return [b for b in batch]

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )


def generate_calibrate_set(
    model,
    sampler,
    sample_dataloader,
    calib_layer,
    dump_path,
    chunk_size=50,
    compress=True,
    **kwargs
):
    dump = True
    if dump_path is None:
        print('[Warning] Disable dump, memory may be overflow')
        dump = False

    if dump:
        if not os.path.exists(dump_path):
            print(f'[INFO] calibset [{dump_path}] not found, generating...')
            os.makedirs(dump_path, exist_ok=True)
        else:
            print(f'[INFO] calibset [{dump_path}] found, loading...')
            return BaseCalibSet(folder=dump_path, compress=compress)
    
    buffer = []
    chunk_idx = 0
    total_samples = 0

    for sample_data in sampler.sample(model, sample_dataloader, sample_mode='input', sample_layer=calib_layer, **kwargs):
        for single_data in sample_data:
            assert 'input' in single_data
            this_tuple = tuple(single_data['input']['args']) + tuple(single_data['input']['kwargs'].values())
            buffer.append(this_tuple)
            total_samples += 1

            if dump:
                if len(buffer) >= chunk_size:
                    chunk_path = os.path.join(dump_path, f"chunk_{chunk_idx:03d}" + (".pt.zst" if compress else ".pt"))
                    if not compress:
                        torch.save(buffer, chunk_path)
                    else:
                        buffer_io = io.BytesIO()
                        torch.save(buffer, buffer_io, _use_new_zipfile_serialization=False)
                        raw_bytes = buffer_io.getvalue()
                        cctx = zstd.ZstdCompressor(level=9, threads=os.cpu_count())
                        compressed_bytes = cctx.compress(raw_bytes)
                        with open(chunk_path, 'wb') as f:
                            f.write(compressed_bytes)
                    buffer.clear()
                    chunk_idx += 1

    if dump:
        if buffer:
            chunk_path = os.path.join(dump_path, f"chunk_{chunk_idx:03d}" + (".pt.zst" if compress else ".pt"))
            if not compress:
                torch.save(buffer, chunk_path)
            else:
                buffer_io = io.BytesIO()
                torch.save(buffer, buffer_io, _use_new_zipfile_serialization=False)
                raw_bytes = buffer_io.getvalue()
                cctx = zstd.ZstdCompressor(level=9, threads=os.cpu_count())
                compressed_bytes = cctx.compress(raw_bytes)
                with open(chunk_path, 'wb') as f:
                    f.write(compressed_bytes)

        print(f"[INFO] Calibration data saved to {dump_path}, total {total_samples} samples.")
        return BaseCalibSet(folder=dump_path, compress=compress)
    else:
        print(f"[INFO] Calibration data completed, total {total_samples} samples.")
        return BaseCalibSet(data=buffer)


if __name__ == '__main__':
    from dataset.coco.coco_dataset import COCODataset
    from sample.sampler import Q_DiffusionSampler
    from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StableDiffusion3ControlNetModel.from_repo(('../stable-diffusion-3-medium-diffusers', '../SD3-Controlnet-Canny'), device)
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)
    sampler = Q_DiffusionSampler()
    sample_dataloader = dataset.get_dataloader()

    calibset = generate_calibrate_set(model, sampler, sample_dataloader, 
                                        sample_layer='dit', 
                                        max_timestep=30, 
                                        sample_size=1, 
                                        timestep_per_sample=1, 
                                        controlnet_conditioning_scale=0.7,
                                        guidance_scale=3.5)
    
    calib_loader = calibset.get_dataloader(batch_size=1, shuffle=True)
    for i, batch in enumerate(calib_loader):
        print('i', i)
        print(len(batch[0]))
        print(batch)
    
    calibset.dump('ca.pt')
