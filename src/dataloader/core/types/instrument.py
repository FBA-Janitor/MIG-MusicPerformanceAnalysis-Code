from typing import Iterable, Literal, Tuple, Union
from enum import Enum


FluteType = Literal["Piccolo", "Flute"]
OboeType = Literal["Oboe", "English Horn"]
BassoonType = Literal["Bassoon", "Contrabassoon"]
DoubleReedType = Union[OboeType, BassoonType]

UpperClarinetsType = Literal["Eb Clarinet", "Bb Clarinet"]
LowerClarinetsType = Literal[
    "Eb Contra Alto Clarinet", "Bass Clarinet", "Bb Contrabass Clarinet"
]
ClarinetType = Union[UpperClarinetsType, LowerClarinetsType]

UpperSaxType = Literal["Soprano Sax", "Alto Saxophone"]
LowerSaxType = Literal["Tenor Saxophone", "Bari Saxophone"]
SaxType = Union[UpperSaxType, LowerSaxType]

UpperWoodwindType = Union[FluteType, OboeType, UpperClarinetsType, UpperSaxType]
LowerWoodwindType = Union[BassoonType, LowerClarinetsType, LowerSaxType]

WoodwindType = Union[UpperWoodwindType, LowerWoodwindType]

UpperBrassType = Literal["Trumpet", "French Horn"]
TromboneType = Literal["Trombone", "Bass Trombone"]
LowBrassType = Union[TromboneType, Literal["Euphonium", "Tuba"]]

BrassType = Union[UpperBrassType, LowBrassType]


InstrumentType = Union[
    WoodwindType,
    BrassType,
    Literal[
        "Piano",
        "Percussion",
    ],
]
