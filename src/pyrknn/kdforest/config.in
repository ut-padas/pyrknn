import os


PYRKNN_USE_CUDA = $pyrknn_cuda
PYRKNN_NUMBA_THREADS = $pyrknn_numba_threads

if os.getenv("KMP_AFFINITY") is None:
    os.environ["KMP_AFFINITY"] = "compact"

if os.getenv("GSKNN_IC_NT") is None:
    os.environ["GSKNN_IC_NT"] = "2"


