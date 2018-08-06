import sys

from snippets.scaffold import get_gpu_metrics
from snippets.scheduler import IterGPUSubprocessScheduler
import unittest
import subprocess
import logging

logging.basicConfig(
    stream=sys.stdout,
    # level="INFO",
)


class TestIterGOUSubprocessScheduler(unittest.TestCase):
    def test_all(self):
        try:
            scheduler = IterGPUSubprocessScheduler("all")
            self.assertEqual(len(scheduler._device2job), len(get_gpu_metrics()))
        except RuntimeError:
            pass

    def test_run(self):
        scheduler = IterGPUSubprocessScheduler(devices=[0, 1, 2, 3], interval=0.001)
        future1 = scheduler.submit("ls /tmp -al", stdout=subprocess.PIPE, shell=True)
        scheduler.submit("cd /tmp && cd ..", shell=True)
        scheduler.submit("cd /tmp && cd ..", shell=True)
        scheduler.submit("cd /tmp && cd ..", shell=True)
        scheduler()
        self.assertTrue(future1["stdout"] is not None)
        self.assertTrue(future1["stderr"] is None)
        future1["stdout"].close()

    def test_failed(self):
        scheduler = IterGPUSubprocessScheduler(devices=[0, 1, 2, 3], interval=0.001)
        future1 = scheduler.submit("cd /data", stderr=subprocess.PIPE, shell=True, env={})
        scheduler()
        self.assertTrue(future1["stdout"] is None)
        self.assertTrue(future1["stderr"] is not None)
        future1["stderr"].close()
        self.assertEqual(scheduler.failed_count, 1)

    def test_restart(self):
        scheduler = IterGPUSubprocessScheduler(devices=[0, 1, 2, 3], interval=0.001, restart_failed=True)
        future1 = scheduler.submit(" if [ ! -f /tmp/foo.txt ]; then touch /tmp/foo.txt && exit 1; \
                    else rm /tmp/foo.txt && exit 0; fi",
                                   stderr=subprocess.PIPE, shell=True, env={})
        scheduler()
        self.assertTrue(future1["stdout"] is None)
        self.assertTrue(future1["stderr"] is not None)
        future1["stderr"].close()
        self.assertEqual(scheduler.failed_count, 0)
        self.assertEqual(scheduler._failed_job_indices, set())
        self.assertEqual(scheduler._next_start_job_index, 1)
