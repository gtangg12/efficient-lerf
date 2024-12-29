from __future__ import annotations

import torch
import torch.nn.functional as F

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence


def sequence_downsample(sequence: FrameSequence, downsample: int) -> FrameSequence:
    """
    """
    sequence = sequence.clone()
    scale = 1 / downsample
    sequence.rescale_camera_resolution(scale=scale)
    images = sequence.images.permute(0, 3, 1, 2)
    images = F.interpolate(images, scale_factor=scale, mode='bilinear')
    sequence.images = images.permute(0, 2, 3, 1)
    return sequence


if __name__ == '__main__':
    pass