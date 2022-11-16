from core.subdatasets.segment import SegmentDataset
from core.subdatasets.audio import AudioDataset
from core import GenericDataset
    
    
class FBADataset(GenericDataset):
    def __init__(
        self,
        student_information,
        use_audio=True,
        use_f0=False,
        segment=False,
        score=True,
        config_root="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config",
        audio_data_root="/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/audio/bystudent",
        feature_data_root="/home/yding402/fba-data/tmp_feature",
        segment_data_root="/home/yding402/fba-data/MIG-FBA-Data-Cleaning/cleaned/segmentation/bystudent"
    ) -> None:
        super().__init__()

        self.student_information = student_information
        self.student_ids = [student_id for student_id, year, band in student_information]
        self.length = len(self.student_ids)

        self.subdatasets = dict()
        if use_audio:
            self.subdatasets["audio"] = AudioDataset(student_information, data_root=audio_data_root, feature_save_dir=feature_data_root)
        if segment:
            self.subdatasets["segment"] = SegmentDataset(student_information, data_root=segment_data_root)

    def get_item_by_student_id(self, sid):
        data = {key: dataset.get_item_by_student_id(sid) for key, dataset in self.subdatasets.items()}
        data["student_id"] = sid
        return data

    def get_item_by_index(self, idx):
        return self.get_item_by_student_id(self.student_ids[idx])

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item_by_index(index)
        elif isinstance(index, str):
            return self.get_item_by_student_id(index)
        else:
            raise KeyError("Invalid index {}".format(index))

    def __len__(self):
        return self.length
