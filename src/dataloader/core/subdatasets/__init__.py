from abc import ABC


class GenericSubdataset(ABC):
    def __init__(self, student_ids) -> None:
        self.student_ids = student_ids

    def get_index_from_student_id(self, sid):
        return self.student_ids.index(sid)

    def get_item_by_index(self, idx):
        pass

    def get_item_by_student_id(self, sid):
        return self.get_item_by_index(self.get_index_from_student_id(sid))
