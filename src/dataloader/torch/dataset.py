from typing import Dict
from torch.utils import data
from ..core.dataset import FBADataset as FBADatasetCore

class FBADataset(data.Dataset):
    def __init__(
        self,
        dataset_kwargs: Dict,
    ) -> None:
        super().__init__()

        self.dataset = FBADatasetCore(**dataset_kwargs)

    def __getitem__(self, index):
        return self.dataset.get_item_by_index(index)

    def __len__(self):
        return self.dataset.length
