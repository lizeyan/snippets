import time
import typing
from collections import Iterable
from contextlib import contextmanager
from typing import List, Dict, Union, Any
import warnings

import numpy as np
from torch.utils.data import DataLoader

from .metric import Metric
from ..utilities import assert_positive_integer


class Loop(object):
    """
    Train Loop
    Example Usage:
        with TrainLoop().with_context() as loop:
            for epoch in loop.iter_epochs():
                for step, data in loop.iter_steps(data_loader):
                    compute code
                    loop.submit_metric(name, metric value)
                    loop.submit_data(name, data value)
        loop.get_metric(epoch)
        # dict, key is the metric name, value is the list of the values submitted in that epoch
        loop.get_metric(name)
        # tuple of lists,
        # the first one is the list of epochs that contain this metric,
        # the second one is the list of metrics, each element is a list of submitted metric values
        # data are just all the same, but they won't displayed in screen
        loop.get_data(epoch)
        loop.get_data(name)
    """
    __EPOCH_TIME_KEY = "epoch_time(s)"
    __STEP_TIME_KEY = "step_time(s)"

    def __init__(self, max_epochs: typing.Union[int, None] = None,
                 max_steps: typing.Union[int, None] = None,
                 disp_step_freq=None,
                 disp_epoch_freq=1,
                 print_fn: Union[typing.Callable[[str], None], None] = print,):
        """
        :param max_epochs: Either max_epochs or max_steps should be a valid value
        :param max_steps: Either max_epochs or max_steps should be a valid value
        :param disp_epoch_freq: the interval (counting in epochs) between two logging messages
        :param print_fn: the function used to print logging messages
        """

        assert max_epochs is not None or max_steps is not None, \
            "At least one of max_epochs and max_steps should not be None"
        assert max_epochs is None or assert_positive_integer(max_epochs=max_epochs)
        assert max_steps is None or assert_positive_integer(max_steps=max_steps)
        assert disp_epoch_freq is None or assert_positive_integer(disp_epoch_freq=disp_epoch_freq)
        assert disp_step_freq is None or assert_positive_integer(disp_step_freq=disp_step_freq)
        self._max_epochs = max_epochs
        self._max_steps = max_steps
        self._print_fn = print_fn
        self._disp_epoch_freq = disp_epoch_freq
        self._disp_step_freq = disp_step_freq

        self._epoch_cnt = 0  # type: int
        self._step_cnt = 0  # type: int
        self._displayed_at_epoch = 0  # type: int
        self._displayed_at_step = 0  # type: int

        self._metrics = {}  # type: Dict[str, Metric]
        self._data = {}  # type: Dict[str, Metric]

        self._within_epochs = False
        self._within_steps = False
        self._within_context = False

        self._epoch2step_dict = {}  # Dict[int, List]
        self._step2epoch_dict = {}  # Dict[int, int]

    @contextmanager
    def with_context(self):
        """
        context manager for Loop
        :return:
        """
        self._within_context = True
        yield self
        self._within_context = False

    def _eta(self):
        try:
            epoch_time_estimate = np.asscalar(np.mean(self._metrics[self.__EPOCH_TIME_KEY]["all"]))
        except KeyError:
            epoch_time_estimate = float("inf")
        try:
            step_time_estimate = np.asscalar(np.mean(self._metrics[self.__STEP_TIME_KEY]["all"]))
        except KeyError:
            step_time_estimate = float("inf")
        estimate = float("inf")
        if self._max_epochs is not None:
            estimate = min(estimate, (self._max_epochs - self._epoch_cnt) * epoch_time_estimate)
        if self._max_steps is not None:
            estimate = min(estimate, (self._max_steps - self._step_cnt) * step_time_estimate)
        return estimate

    def iter_epochs(self):
        def loop_condition():
            return (self._max_epochs is None or self._epoch_cnt < self._max_epochs) and (
                    self._max_steps is None or self._step_cnt < self._max_steps)

        def disp_condition():
            return self._disp_epoch_freq is not None and \
                   self._epoch_cnt % self._disp_epoch_freq == 0

        assert self._within_context, "iter_epochs() should be called in context manager"

        self._within_epochs = True

        try:
            while loop_condition():
                self._epoch_cnt += 1
                self._epoch2step_dict[self._epoch_cnt] = []
                tic = time.time()
                remember_step_count = self._step_cnt
                yield self._epoch_cnt
                # a epoch should contains at least one step
                if remember_step_count == self._step_cnt:
                    warnings.warn("An epoch should contains at least one step")
                    self._step_cnt += 1
                    self._epoch2step_dict[self._epoch_cnt].append(self._step_cnt)
                    self._step2epoch_dict[self._step_cnt] = self._epoch_cnt

                toc = time.time()
                self.submit_metric(self.__EPOCH_TIME_KEY, toc - tic)
                if disp_condition():
                    self._print_log(unit="epoch")
                    self._displayed_at_epoch = self._epoch_cnt
        finally:
            self._within_epochs = False

    def iter_steps(self, dataloader: Union[DataLoader, Iterable]):
        assert self._within_context, "iter_epochs() should be called in context manager"
        assert self._within_epochs, "iter_steps() should be called in an iter_epoch."
        self._within_steps = True

        def disp_condition():
            return self._disp_step_freq is not None and \
                   self._step_cnt % self._disp_step_freq == 0
        try:
            for data in dataloader:
                self._step_cnt += 1
                self._epoch2step_dict[self._epoch_cnt].append(self._step_cnt)
                self._step2epoch_dict[self._step_cnt] = self._epoch_cnt
                tic = time.time()
                yield self._step_cnt, data
                toc = time.time()
                self.submit_metric(self.__STEP_TIME_KEY, toc - tic)
                if self._max_steps is not None and self._step_cnt >= self._max_steps:
                    break
                if disp_condition():
                    self._print_log(unit="step")
                    self._displayed_at_step = self._step_cnt
        finally:
            self._within_steps = False

    def epoch2step(self, item):
        if isinstance(item, int):
            return self._epoch2step_dict[item]
        else:
            return list(sum(self._epoch2step_dict[_] for _ in item))

    def _get(self, data, name, *, step=None, epoch=None):
        assert step is None or epoch is None, "step and epoch can be not None either"
        if epoch is not None and step is None:
            step = self.epoch2step(epoch)
        if step is None:
            return data[name]
        else:
            return data[name][step]

    def get_metric(self, name, *, step=None, epoch=None):
        return self._get(self._metrics, name, step=step, epoch=epoch)

    def get_data(self, name, *, step=None, epoch=None):
        return self._get(self._data, name, step=step, epoch=epoch)

    def submit_metric(self, name, value):
        if self._within_steps or self._within_epochs:
            if name not in self._metrics:
                self._metrics[name] = Metric(name)
            self._metrics[name].collect(self._step_cnt, value)
        else:
            raise RuntimeError("Can't submit metric outside epoch or step")

    def submit_data(self, name, value):
        if self._within_steps or self._within_epochs:
            if name not in self._data:
                self._data[name] = Metric(name)
            self._data[name].collect(self._step_cnt, value)
        else:
            raise RuntimeError("Can't submit data outside epoch or step")

    def print(self, string):
        if self._max_epochs is None:
            epoch_str = "{}".format(self._epoch_cnt)
        else:
            epoch_str = "{}/{}".format(self._epoch_cnt, self._max_epochs)
        if self._max_steps is None:
            step_str = "{}".format(self._step_cnt)
        else:
            step_str = "{}/{}".format(self._step_cnt, self._max_steps)
        process_str = "[epoch:{} step:{} ETA:{:.3f}s]".format(epoch_str, step_str,
                                                              self._eta())
        self._print_fn("{} {}".format(process_str, string))

    def _print_log(self, unit: str):
        if self._print_fn is None:
            return
        if unit == "step":
            item = np.arange(self._displayed_at_step + 1, self._step_cnt + 1)
        elif unit == "epoch":
            item = np.concatenate([self._epoch2step_dict[_]
                                   for _ in np.arange(self._displayed_at_epoch + 1, self._epoch_cnt + 1)])
        else:
            raise ValueError(f"Unknown unit: {unit}")
        metric_str_list = []
        for name, metric in self._metrics.items():
            metric_str_list.append(metric.format(item))
        metric_str = " ".join(metric_str_list)

        self.print(metric_str)


TrainLoop = Loop


class TestLoop(Loop):
    """
    A subclass of Loop.
    There is not "iter_epochs" in TestLoop.
    And operations will not compute grads by default
    """
    def __init__(self,
                 max_steps: typing.Union[int, None] = None,
                 disp_step_freq=None,
                 print_fn: Union[typing.Callable[[str], None], None] = print,
                 no_grad: bool=True):
        """
        :param print_fn: the print function
        :param no_grad: disable computing grads for pytorch operations or not
        """
        super(TestLoop, self).__init__(max_epochs=1, max_steps=max_steps,
                                       disp_epoch_freq=1, disp_step_freq=disp_step_freq,
                                       print_fn=print_fn,
                                       )
        self.no_grad = no_grad

    def iter_epochs(self):
        raise RuntimeError("TestLoop don't need to iterate epochs.")

    @contextmanager
    def with_context(self):
        """
        context manager for TestLoop
        :return:
        """
        import torch
        self._within_context = True
        for _ in super().iter_epochs():
            if self.no_grad:
                with torch.no_grad():
                    yield self
            else:
                yield self
        self._within_context = False
