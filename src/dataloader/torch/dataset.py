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
        item = self.dataset.get_item_by_index(index)
        # print({
        #     key: value.shape for key, value in item.items() if "shape" in dir(value)
        # })
        return item

    def __len__(self):
        return self.dataset.length
