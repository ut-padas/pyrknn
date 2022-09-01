import os


PYRKNN_USE_CUDA = False
PYRKNN_NUMBA_THREADS = 8

if os.getenv("KMP_AFFINITY") is None:
    os.environ["KMP_AFFINITY"] = "compact"

if os.getenv("GSKNN_IC_NT") is None:
    os.environ["GSKNN_IC_NT"] = "2"


