from subprocess import *
import pandas as pd

_QUERIES = ["index", "gpu_name", "utilization.gpu",
            "utilization.memory", "memory.free", "memory.total", "driver_version"]
_COMMAND = f"nvidia-smi --query-gpu={','.join(__QUERIES)} --format=csv,nounits"


def get_gpu_metrics() -> pd.DataFrame:
    smi = Popen(_COMMAND, shell=True, stdout=PIPE)
    stdout = smi.communicate(timeout=5)[0]
    df = pd.read_csv(stdout, index_col=None)
    return df


def sort_gpu_index() -> list:
    df = get_gpu_metrics()
