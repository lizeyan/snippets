from typing import *
import time
import logging
from subprocess import Popen
from ..scaffold import get_gpu_metrics


class IterGPUSubprocessScheduler(object):
    def __init__(self, devices: Union[str, Iterable[int]]="all", interval: float=1):
        if devices == "all":
            devices = list(int(_) for _ in get_gpu_metrics()["index"])
        self._device2job = {device: None for device in devices}
        assert len(self._device2job) > 0, "No GPU device given"
        self._interval = interval

        self._job_args_list = []
        self._next_start_job_index = 0
        self._available_device = list(self._device2job.keys())[0]
        self._future_list = []
        self._finished_cnt = 0

        self.__called = False

    def submit(self, *args, **kwargs):
        self._job_args_list.append((args, kwargs))
        future = {}
        self._future_list.append(future)
        return future

    def __call__(self, *args, **kwargs):
        assert not self.__called, "It can't be called again, create a new object"
        while self._has_unfinished_jobs() or self._has_pending_jobs():
            while self._has_avail_device() and self._has_pending_jobs():
                self._start_job()
            time.sleep(self._interval)
        self.__called = True

    def _has_unfinished_jobs(self) -> bool:
        self._update_device_states()
        return any(_ is not None for _ in self._device2job.values())

    def _update_device_states(self):
        for key in self._device2job.keys():
            if self._device2job[key] is None:
                continue
            job = self._device2job[key]["job"]
            index = self._device2job[key]["index"]
            setting = self._job_args_list[index]
            if job.poll() is not None:
                logging.getLogger(__file__).info(f"job at device {key} exits with {job.poll()}: {setting}")
                self._finished_cnt += 1
                logging.getLogger(__file__).info(f"{self._finished_cnt}/{len(self._job_args_list)} ended")
                self._future_list[index]["stdout"] = job.stdout
                self._future_list[index]["stderr"] = job.stderr
                self._device2job[key] = None

    def _has_pending_jobs(self) -> bool:
        return self._next_start_job_index is not None and self._next_start_job_index < len(self._job_args_list)

    def _has_avail_device(self):
        self._update_device_states()
        for key in self._device2job.keys():
            if self._device2job[key] is None:
                self._available_device = key
                return True
        return False

    def _start_job(self):
        assert self._available_device is not None and self._next_start_job_index < len(self._job_args_list)
        args, kwargs = self._job_args_list[self._next_start_job_index]
        job = Popen(*args, **kwargs)
        self._device2job[self._available_device] = {"job": job, "index": self._next_start_job_index}
        logging.getLogger(__file__).info(f"job assigned to {self._available_device}: {args, kwargs}")
        self._available_device = None
        self._next_start_job_index += 1
        logging.getLogger(__file__).info(f"{self._next_start_job_index}/{len(self._job_args_list)} started")

