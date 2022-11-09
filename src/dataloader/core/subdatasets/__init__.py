from abc import ABC


class GenericSubdataset(ABC):
    def __init__(
        self,
        student_information,
        data_root,
        ) -> None:

        self.student_information = student_information
        self.student_ids = [student_id for student_id, year, band in student_information]

        self.data_root = data_root
        self.data_path = {sid: None for sid in self.student_ids}

        self._load_data_path()

    def get_index_from_student_id(self, sid):
        return self.student_ids.index(sid)

    def get_item_by_student_id(self, sid):
        return self.read_data_file(self.data_path[str(sid)])

    def get_item_by_index(self, idx):
        return self.get_item_by_student_id(self.student_ids[idx])

    def _load_data_path(self):
        raise NotImplementedError("Must be implemented in the Subdataset")

    def read_data_file(self):
        raise NotImplementedError("Must be implemented in the Subdataset")
