from subprocess import *
import pandas as pd
import os

_QUERIES = ["index", "gpu_name", "utilization.gpu",
            "utilization.memory", "memory.free", "memory.total", "driver_version"]
_COMMAND = f"nvidia-smi --query-gpu={','.join(_QUERIES)} --format=csv,nounits"


def get_gpu_metrics() -> pd.DataFrame:
    """
    :return: pandas.DataFrame, contains GPU index, name, utilization, memory and driver information
    """
    smi = Popen(_COMMAND, shell=True, stdout=PIPE, env=os.environ)
    smi.wait()
    if smi.poll() != 0:
        df = pd.DataFrame()
    else:
        df = pd.read_csv(smi.stdout, index_col=None)
    smi.stdout.close()
    return df


def sort_gpu_index() -> list:
    """
    :return: return a list of GPU index, sorted by free memory (descending) and utilization (ascending)
    """
    df = get_gpu_metrics()
    if len(df) == 0:
        return []
    else:
        df = df.sort_values(by=[" memory.free [MiB]", " utilization.gpu [%]"], ascending=[0, 1])
        return list(int(i) for i in df.index)
