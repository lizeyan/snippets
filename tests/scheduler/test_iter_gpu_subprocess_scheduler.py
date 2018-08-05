from snippets.scheduler import IterGPUSubprocessScheduler
import unittest
import subprocess


class TestIterGOUSubprocessScheduler(unittest.TestCase):
    def test(self):
        scheduler = IterGPUSubprocessScheduler(devices="all", interval=0.001)
        future1 = scheduler.submit("ls /tmp -al", stdout=subprocess.PIPE, shell=True)
        scheduler()
        self.assertTrue(future1["stdout"] is not None)
        self.assertTrue(future1["stderr"] is None)
        future1["stdout"].close()
