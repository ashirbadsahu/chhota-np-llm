import numpy as np

HAS_GPU = False
DEVICE = "CPU (NumPy)"
xp = np

try:
    import cupy as _cp
    _ = _cp.zeros((1,))
    _ = _cp.random.randn(1)
    HAS_GPU = True
    xp = _cp
    DEVICE = "GPU (CuPy)"
    _cp.get_default_memory_pool().free_all_blocks()
except Exception:
    HAS_GPU = False
    xp = np
    DEVICE = "CPU (NumPy)"


def as_numpy(arr):
    if HAS_GPU and hasattr(arr, "get"):
        return arr.get()
    if isinstance(arr, np.ndarray):
        return arr
    return np.array(arr)


def as_device(arr):
    if HAS_GPU and isinstance(arr, np.ndarray):
        return xp.asarray(arr)
    return arr


def print_device():
    print(f"Using: {DEVICE}")
