from typing import Iterable, Literal, Tuple, Union, _LiteralGenericAlias
from enum import Enum


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

# TODO: Percussion Segment Type
# PercussionSegmentType = Literal[""]
# SegmentType = Union[WindSegmentType, PercussionSegmentType]

