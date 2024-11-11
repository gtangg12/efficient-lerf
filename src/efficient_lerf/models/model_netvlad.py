from pathlib import Path

from natsort import natsorted
from hloc import extract_features
from hloc.pairs_from_retrieval import *

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.utils.math import norm


class ModelNetVLAD:
    """
    """
    def __init__(self, method='netvlad'):
        """
        """
        self.method = method
        self.config = extract_features.confs[method]

    def __call__(self, sequence: FrameSequence) -> TorchTensor["N", "dim"]:
        """
        """
        path = sequence.metadata['data_dir']
        filename = extract_features.main(self.config, path / 'images', export_dir=path)
        names = list_h5_names(filename)
        names = natsorted(names)
        return norm(get_descriptors(names, filename), dim=-1)


if __name__ == '__main__':
    pass