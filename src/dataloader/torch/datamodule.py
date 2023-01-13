import os
from typing import List, Tuple, Optional
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from .dataset import FBADataset


class FBADataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_yearbands: List[Tuple(int, str)],
        val_yearbands: Optional[List[Tuple(int, str)]] = None,
        split_path="/media/fba/ProbabilisticMPA/MIG-MusicPerformanceAnalysis-Code/src/split",
    ) -> None:
        super().__init__()

        self.split_path = split_path

        if val_yearbands is None:
            val_yearbands = train_yearbands

        self.train_ids = self.get_split_ids("train", train_yearbands)
        self.val_ids = self.get_split_ids("valtest", val_yearbands)

    def get_split_ids(self, split, yearbands):
        ids = []

        for y, b in yearbands:
            id = np.load(
                os.path.join(self.split_path, "canonical", str(y), b, f"{split}.npy")
            )
            ids.append(id)

        ids = np.concatenate(ids)

        return ids

    def train_dataloader(self):
        return DataLoader(
            FBADataset(self.train_ids), batch_size=32, shuffle=True, num_workers=4
        )
