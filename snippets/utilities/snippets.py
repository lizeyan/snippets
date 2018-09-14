from typing import *


def in_same_length(*args) -> bool:
    if len(args) == 0:
        return True
    length = len(args[0])
    for ele in args[1:]:
        if len(ele) != length:
            return False
    return True


def split(x: Sequence, split_radio: List[float]):
    total_length = len(x)
    assert 0. <= sum(split_radio) <= 1. and min(split_radio) > 0., "split radio is illegal"
    ptr = 0
    ret = []
    for radio in split_radio:
        new_ptr = ptr + int(total_length * radio)
        ret.append(x[ptr: new_ptr])
        ptr = new_ptr
    return tuple(ret)
