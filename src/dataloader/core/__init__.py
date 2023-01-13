from abc import ABC


class GenericDataset(ABC):
    def __init__(self) -> None:
        pass
    
    def __getitem__(self, index) -> None:
        pass
    
    def __len__(self):
        pass
    
    def as_numpy(self):    
        pass