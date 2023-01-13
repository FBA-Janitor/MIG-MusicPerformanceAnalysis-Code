from core import GenericDataset
from typing import List

class PitchDataset(GenericDataset):
    def __init__(self, student_ids: List[str], data_root, file_name_format) -> None:
        super().__init__()
        
        self.student_ids = student_ids
        
    def get_item_by_index(self, idx):
        pass
    
    def get_item_by_student_id(self, sid):
        return self.get_item_by_index(self.student_ids.index(sid))
    