from torch.utils import data

class FBADataset(data.Dataset):
    def __init__(
        self,
        student_ids,
        use_audio,
        use_f0,
        config_root="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config",
    ) -> None:
        super().__init__()

        self.length = len(student_ids)
        self.dataset = None

    def __getitem__(self, index):
        return super().__getitem__(index)

    def __len__(self):
        return self.length
