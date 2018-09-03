from contextlib import contextmanager
from typing import Union, Iterable, Callable
import numpy as np
from collections import OrderedDict


class Metric(object):
    _IGNORE_KEY_ERROR = True

    def __init__(self, name: str):
        self.name = name
        self.data = OrderedDict()  # Dict[int, Any]
        self._step_count = {}

    def collect(self, global_step: int, value):
        if global_step not in self._step_count:
            self._step_count[global_step] = 0
        self._step_count[global_step] += 1
        if self.name == "test_loss":
            pass
        if global_step not in self.data:
            self.data[global_step] = value
        else:
            if self._step_count[global_step] == 2:
                self.data[global_step] = [self.data[global_step]]
            self.data[global_step].append(value)

    def __getitem__(self, item: Union[int, Iterable[int], None]):
        if isinstance(item, str) and item == "last":
            return list(self.data.values())[-1]
        elif isinstance(item, int):
            try:
                return self.data[item]
            except KeyError as e:
                if not self._IGNORE_KEY_ERROR:
                    raise e
        elif isinstance(item, str) and item == "all":
            return list(self.data.values())
        else:
            ret = []
            for _ in item:
                try:
                    ret.append(self.data[_])
                except KeyError as e:
                    if not self._IGNORE_KEY_ERROR:
                        raise e
            return ret

    def format(self, item: Union[int, Iterable[int]]=None,
               number_precision=".3f",
               mean: Callable=np.mean, std: Callable=np.std):
        data = self[item] if item is not None else list(self.data.values())
        if np.size(data) > 1:
            return f"{self.name}:{mean(data):{number_precision}}(±{std(data):{number_precision}})"
        elif np.size(data) == 1:
            return f"{self.name}:{mean(data):{number_precision}}"
        else:
            return ""

    @staticmethod
    @contextmanager
    def raise_key_error():
        Metric._IGNORE_KEY_ERROR = False
        yield
        Metric._IGNORE_KEY_ERROR = True
