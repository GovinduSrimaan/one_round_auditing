"""base.py — shared constants and seed helpers.

Changes vs original
--------------------
* DEVICE now auto-selects CUDA → MPS → CPU so the code runs on any machine.
* Added ``get_device()`` helper for explicit use where needed.
* No other logic changed.
"""

import os
import random
import typing

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Device / dtype
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()
DTYPE = torch.float32
DTYPE_EVAL = torch.float64
EVAL_BATCH_SIZE = 1024


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def setup_seeds(
    seed: int,
    deterministic_algorithms: bool = True,
    benchmark_algorithms: bool = False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

        if benchmark_algorithms:
            raise ValueError("Benchmarking should not be enabled under deterministic algorithms")

    torch.backends.cudnn.benchmark = benchmark_algorithms


def get_setting_seed(
    global_seed: int,
    shadow_model_idx: typing.Optional[int],
    num_shadow: int,
) -> int:
    return global_seed * (num_shadow + 1) + (0 if shadow_model_idx is None else shadow_model_idx + 1)
