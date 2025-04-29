from typing import Union
import torch
from torch.utils.data import Dataset
from backend.torch.models.stable_diffusion_3_controlnet import StableDiffusion3ControlNetModel
from backend.torch.models.flux_controlnet import FluxControlNetModel
from sample.sampler import BaseSampler

class BaseCalibSet(Dataset):
    @classmethod
    def from_file(cls, path: str):
        """Load dataset from a .pt file"""
        if not path.endswith('.pt'):
            raise ValueError("Only .pt format is supported for load.")
        data = torch.load(path)
        return cls(data)

    @staticmethod
    def collate_fn(batch):
        return [b for b in batch]

    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x
    
    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )

    def dump(self, path: str):
        """Save the dataset to a .pt file"""
        if not path.endswith('.pt'):
            raise ValueError("Only .pt format is supported for dump.")
        torch.save(self.data, path)

def generate_calibrate_set(model: Union[StableDiffusion3ControlNetModel, FluxControlNetModel], sampler: BaseSampler, sample_dataloader, calib_layer: str, **kwargs):    
    data = []
    for sample_data in sampler.sample(model, sample_dataloader, sample_mode='input', sample_layer=calib_layer, **kwargs):
        for single_data in sample_data:
            assert 'input' in single_data
            this_tuple = tuple(single_data['input']['args']) + tuple(single_data['input']['kwargs'].values())
            data.append(this_tuple)
    
    return BaseCalibSet(data)
        


if __name__ == '__main__':
    from dataset.coco.coco_dataset import COCODataset
    from sample.sampler import Q_DiffusionSampler

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
