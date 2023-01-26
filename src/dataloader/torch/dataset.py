from typing import Dict
from torch.utils import data
from ..core.dataset import FBADataset as FBADatasetCore
from ..core.types.segment import SegmentType


class FBADataset(data.Dataset):
    def __init__(
        self,
        student_information,
        dataset_kwargs: Dict,
    ) -> None:
        super().__init__()

        self.dataset = FBADatasetCore(
            student_information=student_information, **dataset_kwargs
        )

    def __getitem__(self, index):
        return self.dataset.get_item_by_index(index)

    def __len__(self):
        return self.dataset.length
