from typing import *
import time
import logging
from subprocess import Popen
import os
from ..scaffold import get_gpu_metrics


class IterGPUSubprocessScheduler(object):
    def __init__(self, devices: Union[str, Iterable[int]]="all",
                 interval: float=1,
                 restart_failed: bool=False):
        if devices == "all":
            devices = list(int(_) for _ in get_gpu_metrics()["index"])
        self._device2job = {device: None for device in devices}
        assert len(self._device2job) > 0, "No GPU device given"
        self._interval = interval
        self._restart_failed = restart_failed

        self._job_args_list = []
        self._next_start_job_index = 0
        self._available_device = list(self._device2job.keys())[0]
        self._future_list = []
        self._finished_cnt = 0
        self._failed_job_indices = set()

        self.__called = False

    @property
    def failed_count(self):
        assert self.__called, "This scheduler has not run."
        return len(self._failed_job_indices)

    def submit(self, *args, **kwargs):
        self._job_args_list.append((args, kwargs))
        future = {}
        self._future_list.append(future)
        return future

    def __call__(self, *args, **kwargs):
        assert not self.__called, "It can't be called again, create a new object"
        while self._has_unfinished_jobs() or self._has_pending_jobs():
            while self._has_avail_device() and self._has_pending_jobs():
                self._start_next_job()
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
                logging.getLogger(__file__).info(f"job {index} at device {key} exits with {job.poll()}: {setting}")
                self._finished_cnt += 1
                self._future_list[index]["stdout"] = job.stdout
                self._future_list[index]["stderr"] = job.stderr
                self._device2job[key] = None
                if job.poll() is not 0:
                    self._failed_job_indices.add(index)
                    if self._restart_failed:
                        job.stdout.close() if job.stdout is not None else None
                        job.stderr.close() if job.stderr is not None else None
                        self._restart_job(index, key)
                else:
                    if index in self._failed_job_indices:
                        self._failed_job_indices.remove(index)

    def _has_pending_jobs(self) -> bool:
        return self._next_start_job_index is not None and self._next_start_job_index < len(self._job_args_list)

    def _has_avail_device(self):
        self._update_device_states()
        for key in self._device2job.keys():
            if self._device2job[key] is None:
                self._available_device = key
                return True
        return False

    def _start_next_job(self):
        assert self._available_device is not None and self._next_start_job_index < len(self._job_args_list)
        self._start_job(self._next_start_job_index, self._available_device)
        logging.getLogger(__file__).info(f"{self._next_start_job_index + 1}/{len(self._job_args_list)} started")
        self._next_start_job_index += 1
        self._available_device = None

    def _start_job(self, index, device):
        args, kwargs = self._job_args_list[index]
        if "env" in kwargs:
            env = kwargs["env"]
        else:
            env = os.environ
        env["CUDA_VISIBLE_DEVICES"] = f"{device}"
        kwargs["env"] = env
        job = Popen(*args, **kwargs)
        self._device2job[device] = {"job": job, "index": index}
        logging.getLogger(__file__).info(f"job assigned to {self._available_device}: {args, kwargs}")

    def _restart_job(self, index, key):
        self._start_job(index, key)
        logging.getLogger(__file__).info(f"{index + 1}/{len(self._job_args_list)} restarted")
