import pytorch_lightning as pl


class FBADataModule(pl.LightningDataModule):
    def __init__(self, config_root='/media/fba/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config') -> None:
        super().__init__()        
        
    def prepare_data(self) -> None:
        return super().prepare_data()
    