import os
from typing import List, Tuple, Optional
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from .dataset import FBADataset


class FBADataModule(pl.LightningDataModule):
    def __init__(
        self,
        year_band_inst: List[Tuple[int, str, str]],
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        split_path="/media/fba/ProbabilisticMPA/MIG-MusicPerformanceAnalysis-Code/src/split",
    ) -> None:
        super().__init__()

        self.split_path = split_path

        self.train_ids = self.get_split_ids("train", year_band_inst)
        self.val_ids = self.get_split_ids("valtest", year_band_inst)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

    def get_split_ids(self, split, year_band_inst: List[Tuple[int, str, str]]):
        ids = []

        for y, b, i in year_band_inst:
            id = np.load(
                os.path.join(
                    self.split_path,
                    "canonical",
                    str(y),
                    b,
                    f"{split}-{i.replace(' ', '')}.npy",
                )
            )
            ids.append(id)

        ids = np.concatenate(ids)

        return ids

    def train_dataloader(self):
        return DataLoader(
            FBADataset(self.train_ids),
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            FBADataset(self.val_ids),
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    dm = FBADataModule([(2013, "middle", "Trumpet"), (2013, "middle", "Clarinet")])
