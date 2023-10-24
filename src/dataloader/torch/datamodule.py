import os
from typing import Dict, List, Tuple, Optional
import warnings
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from .dataset import FBADataset
from torch.utils.data._utils.collate import collate, default_collate_fn_map, collate_numpy_array_fn


fba_collate_fn_map = {
    **default_collate_fn_map,
}


def collate_numpy_array_fba_fn(batch, *, collate_fn_map = None):
    if np.array_equal(batch[0].shape[-1] * np.ones(len(batch)), np.array([x.shape[-1] for x in batch])):
        return collate_numpy_array_fn(batch)
    else:
        max_len = max([x.shape[-1] for x in batch])
        return collate_numpy_array_fn(
            [
                np.pad(
                    x, (0, max_len - x.shape[-1]), mode="constant", constant_values=0.0 
                ) for x in batch
            ]
        )



fba_collate_fn_map[np.ndarray] = collate_numpy_array_fba_fn

def fba_collate(batch):
    return collate(batch, collate_fn_map=fba_collate_fn_map)


class FBADataModule(pl.LightningDataModule):
    def __init__(
        self,
        year_band_inst: List[Tuple[int, str, str]],
        dataset_kwargs: Dict = {},
        train_kwargs: Dict = {},
        val_kwargs: Dict = {},
        batch_size: int = 32,
        num_workers: int = 2,
        shuffle_train: bool = True,
        shuffle_val: bool = True,
        split_path="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/split",
        verbose=False,
    ) -> None:
        super().__init__()

        self.split_path = split_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val

        self.dataset_kwargs = dataset_kwargs
        self.train_kwargs = train_kwargs
        self.val_kwargs = val_kwargs

        self.verbose = verbose

        self.train_info = self.get_split_ids("train", year_band_inst)
        self.val_info = self.get_split_ids("valtest", year_band_inst)

        # np.savez("/data/kwatchar3/fba/ProbabilisticMPA/split_info.npz", train=self.train_info, val=self.val_info)

    def get_split_ids(self, split, year_band_inst: List[Tuple[int, str, str]]):
        ids = []

        for y, b, i in year_band_inst:
            load_path = os.path.join(
                self.split_path,
                "canonical",
                str(y),
                b,
                f"{split}-{i.replace(' ', '')}.npy",
            )

            if not os.path.exists(load_path):
                if self.verbose:
                    warnings.warn("No split found for " + load_path)
                continue

            id = np.load(load_path)
            ids.append([(idx, y, b) for idx in id])

        ids = np.concatenate(ids)

        return ids

    def train_dataloader(self):
        return DataLoader(
            FBADataset(self.train_info, {**self.dataset_kwargs, **self.train_kwargs}),
            # collate_fn=fba_collate,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            FBADataset(self.val_info, {**self.dataset_kwargs, **self.val_kwargs}),
            batch_size=self.batch_size,
            shuffle=self.shuffle_val,
            num_workers=self.num_workers,
            drop_last=True,
        )


if __name__ == "__main__":
    dm = FBADataModule(
        [(2013, "middle", "Trumpet"), (2013, "middle", "BbClarinet")],
        use_audio=True,
        use_f0=True,
    )

    for x in dm.train_dataloader():
        print(x)
