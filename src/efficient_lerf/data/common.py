from pathlib import Path

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


CONFIGS_DIR = parent(Path(__file__), 4) / 'configs'