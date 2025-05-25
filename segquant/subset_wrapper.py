from torch.utils.data import Subset, DataLoader


class SubsetWrapper(Subset):
    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
            **kwargs
        )
