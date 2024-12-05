from __future__ import annotations

import torch
import torch.nn.functional as F
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.quantization_methods import compute_superpixels


def downsample(sequence: FrameSequence, downsample: int) -> FrameSequence:
    """
    """
    sequence = sequence.clone()
    scale = 1 / downsample
    sequence.cameras.rescale_output_resolution(scaling_factor=scale)
    images = sequence.images.permute(0, 3, 1, 2)
    images = F.interpolate(images, scale_factor=scale, mode='bilinear')
    sequence.images = images.permute(0, 2, 3, 1)
    return sequence


def upsample(sequence_to_populate: FrameSequence, sequence: FrameSequence) -> FrameSequence:
    """
    """
    assert len(sequence) == len(sequence_to_populate)
    
    upH = sequence_to_populate.cameras[0].height
    upW = sequence_to_populate.cameras[0].width

    sequence_to_populate.clip_codebook_indices = upsample(sequence.clip_codebook_indices, upH, upW)
    sequence_to_populate.dino_codebook_indices = upsample(sequence.dino_codebook_indices, upH, upW)
    sequence_to_populate.clip_codebook = sequence.clip_codebook
    sequence_to_populate.dino_codebook = sequence.dino_codebook
    
    return sequence_to_populate


if __name__ == '__main__':
    pass