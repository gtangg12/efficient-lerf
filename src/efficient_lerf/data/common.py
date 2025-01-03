import os
from glob import glob
from pathlib import Path
from natsort import natsorted

from torchtyping import TensorType


NumpyTensor = TensorType
TorchTensor = TensorType


def parent(path: Path | str, n=1) -> Path | str:
    """ 
    Returns the n-th parent of the given path.
    """
    ispath = isinstance(path, Path)
    path = Path(path)
    for _ in range(n):
        path = path.parent
    return path if ispath else str(path)


# ENVIRONMENT VARIABLES
CONFIGS_DIR = Path(os.environ.get('CONFIGS_DIR', '/home/gtangg12/efficient-lerf/configs'))
DATASET_DIR = Path(os.environ.get('DATASET_DIR', '/home/gtangg12/data'))


def read_datasets(path: Path) -> list[str]:
    return natsorted([Path(path).stem for path in glob(str(path) + '/*')])