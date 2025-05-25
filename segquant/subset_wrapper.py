"""
A wrapper around `torch.utils.data.Subset` that allows access to the original dataset's methods.
"""
from torch.utils.data import Subset, DataLoader


class SubsetWrapper(Subset):
    """
    A wrapper around `torch.utils.data.Subset` that allows access to the original dataset's methods.
    This is useful when you want to create a subset of a dataset but still need to use methods
    from the original dataset, such as `collate_fn`.
    """
    def __getattr__(self, name):
        return getattr(self.dataset, name)

    def get_dataloader(self, batch_size=1, shuffle=False, **kwargs):
        """
        Returns a DataLoader for the subset with the same collate function as the original dataset.
        Args:
            batch_size (int): Size of each batch.
            shuffle (bool): Whether to shuffle the data.
            **kwargs: Additional keyword arguments for DataLoader.
        Returns:
            dataLoader: A DataLoader for the subset.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.dataset.collate_fn,
            **kwargs
        )
