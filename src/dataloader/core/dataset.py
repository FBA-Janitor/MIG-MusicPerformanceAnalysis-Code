from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from .subdatasets import GenericSubdataset
from .subdatasets.segment import SegmentDataset
from .subdatasets.audio import AudioBaseDataset, AudioMelSpecDataset
from .subdatasets.pitch import PitchDataset
from .subdatasets.assessment import AssessmentDataset
from . import GenericDataset

WindSegmentType = Literal[
    "LyricalEtude", "TechnicalEtude", "ChromaticScale", "MajorScales", "SightReading"
]

class WindSegmentEnum(Enum):
    LyricalEtude = 0
    TechnicalEtude = 1
    ChromaticScale = 2
    MajorScales = 3
    SightReading = 4

SegmentType = WindSegmentType

#TODO: Percussion Segment Type
# PercussionSegmentType = Literal[""]
# SegmentType = Union[WindSegmentType, PercussionSegmentType]


class FBADataset(GenericDataset):
    def __init__(
        self,
        student_information,
        segment:SegmentType,
        use_audio=True,
        use_f0=True,
        use_assessment=True,
        config_root="/media/fba/MIG-MusicPerformanceAnalysis-Code/src/data_parse/config",
        assessment_data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/assessment/summary",
        audio_data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/audio/bystudent",
        f0_data_root="/media/fba/MIG-FBA-PitchTracking/cleaned/pitchtrack/bystudent",
        feature_data_root="/media/fba/tmp_audio_feature",
        segment_data_root="/media/fba/MIG-FBA-Data-Cleaning/cleaned/segmentation/bystudent",
        algosegment_data_root="/media/fba/MIG-FBA-Segmentation/cleaned/algo-segmentation/bystudent",
    ) -> None:
        super().__init__()

        self.student_information = student_information
        self.student_ids = [
            student_id for student_id, year, band in student_information
        ]
        self.length = len(self.student_ids)

        self.subdatasets: Dict[str, GenericSubdataset] = dict()

        self.segment_ds = SegmentDataset(
            student_information, data_root=segment_data_root
        )

        self.segment_name = segment
        self.segment_id = WindSegmentEnum[segment].value

        student_information = self.segment_ds.validated_student_information()

        if use_audio:
            self.subdatasets["audio"] = AudioMelSpecDataset(
                student_information,
                data_root=audio_data_root,
            )

        if use_f0:
            self.subdatasets["f0"] = PitchDataset(
                student_information, data_root=f0_data_root, to_midi=True
            )

        if use_assessment:
            self.subdatasets["assessment"] = AssessmentDataset(
                student_information, data_root=assessment_data_root
            )

    def get_item_by_student_id(self, sid):

        start, end = self.segment_ds.get_item_by_student_id(sid)[self.segment_id]

        data = {
            key: dataset.get_item_by_student_id(sid, start, end, self.segment_name)
            for key, dataset in self.subdatasets.items()
        }
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

if __name__ == "__main__":
    
    f0ds = FBADataset(
        student_information=[(29522, 2013, "concert"), (30349, 2013, "concert")],
        segment="LyricalEtude",
    )

    print(f0ds.get_item_by_student_id(29522))


