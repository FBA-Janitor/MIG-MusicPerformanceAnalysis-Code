from .instrument import *
from .segment import *
from .yearband import *

from typing import Literal, Tuple, _LiteralGenericAlias


def get_valid_strings_from_literal(literal_type: Literal):
    args = literal_type.__args__

    strings = []

    for arg in args:
        if type(arg) is _LiteralGenericAlias:
            strings.extend(get_valid_strings_from_literal(arg))
        elif isinstance(arg, Tuple):
            strings.extend(arg)
        elif isinstance(arg, str):
            strings.append(arg)
        else:
            raise ValueError(f"Unknown type {type(arg)}")

    return strings
