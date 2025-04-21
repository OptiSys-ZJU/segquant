import json
from typing import OrderedDict
import torch
from torch.utils.data import Dataset
import os

from backend.torch.utils import load_image

class LimitedCache:
    def __init__(self, max_size):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key, loader_fn):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            value = loader_fn(key)
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
            return value

class COCODataset(Dataset):
    @staticmethod
    def collate_fn(batch):
        return [(prompt, image, control) for prompt, image, control in batch]

    def __init__(self, path, cache_size=1024):
        super().__init__()
        self.base_path = path
        with open(os.path.join(path, 'metadata.json'), "r") as f:
            self.metadata = json.load(f)

        self.image_cache = LimitedCache(cache_size)
        self.control_cache = LimitedCache(cache_size)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        prompt = item["prompt"]
        image_path = os.path.join(self.base_path, item["image"])
        control_path = os.path.join(self.base_path, item["control"])

        image = self.image_cache.get(image_path, load_image)
        control = self.control_cache.get(control_path, load_image)

        return prompt, image, control

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            **kwargs
        )




if __name__ == '__main__':
    dataset = COCODataset(path='../dataset/controlnet_datasets/controlnet_canny_dataset', cache_size=16)
    data_loader = dataset.get_dataloader(batch_size=2)

    for i, batch in enumerate(data_loader):
        prompt, image, control = batch[1]
        print('prompt:', prompt)
        print('image:', image)
        print('control:', control)
        print('--------------')
        if i > 10:
            break
