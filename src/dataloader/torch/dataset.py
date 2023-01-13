from torch.utils import data
from ..core.dataset import FBADataset as CoreFBADataset

class FBADataset(data.Dataset):
    def __init__(
        self,
        **kwargs
    ) -> None:
        super().__init__()

        self.dataset = CoreFBADataset(**kwargs)
        self.length = self.dataset.length

    def __getitem__(self, index):
        return self.dataset.get_item_by_index(index)

    def __len__(self):
        return self.length
