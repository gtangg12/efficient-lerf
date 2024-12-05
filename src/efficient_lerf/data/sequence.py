from __future__ import annotations

import copy
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.math import pad_poses


@dataclass
class FrameSequence:
    """
    """
    cameras: Cameras
    images: TorchTensor['N', 'H', 'W', 3]
    depths: TorchTensor['N', 'H', 'W'] = None

    clip_codebook: TorchTensor['n_clip', 'dim_clip'] = None
    dino_codebook: TorchTensor['n_dino', 'dim_dino'] = None
    clip_codebook_indices: TorchTensor['N', 'M', 'H', 'W'] = None # M is the number of scales
    dino_codebook_indices: TorchTensor['N', 'M', 'H', 'W'] = None # M = 1
    
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.images)
    
    def __getitem__(self, indices: list[int]) -> FrameSequence:
        """
        """
        sequence = {}
        for k, _ in asdict(self).items():
            v = getattr(self, k) # asdict recursively converts non-primitives to dict, including Cameras
            if isinstance(v, torch.Tensor) and len(v) == len(self):
                sequence[k] = v[indices]
            elif isinstance(v, Cameras):
                sequence[k] = v[indices]
            else:
                sequence[k] = v
        return FrameSequence(**sequence)
    
    def clone(self):
        """
        """
        return copy.deepcopy(self)
    
    def transform_cameras(self, scale: float, trans: TorchTensor[4, 4]) -> Cameras:
        """
        """
        cameras = copy.deepcopy(self.cameras)
        cameras.camera_to_worlds = pad_poses(cameras.camera_to_worlds)
        cameras.camera_to_worlds = trans @ cameras.camera_to_worlds
        cameras.camera_to_worlds[:, :3, 3] *= scale
        return cameras
    
    def feature_map(self, name: str, index: int, scale: int = None) -> TorchTensor['H', 'W', 'd']:
        """
        """
        if name == 'clip':
            return self.clip_codebook[self.clip_codebook_indices[index][scale]]
        elif name == 'dino':
            return self.dino_codebook[self.dino_codebook_indices[index][0]]
        raise ValueError(f'Unknown feature map {name}')


def load_sequence(path: Path | str) -> FrameSequence:
    """
    Loads data from disk into a FrameSequence object, overwriting existing data.
    """
    sequence = {}
    for k, _ in FrameSequence.__dataclass_fields__.items():
        object_filename = Path(path) / f'{k}.pt'
        if object_filename.exists():
            with open(object_filename, 'rb') as f:
                sequence[k] = torch.load(f)
            if k == 'cameras':
                sequence[k] = Cameras(**sequence[k])
        else:
            sequence[k] = None
    return FrameSequence(**sequence)


def save_sequence(path: Path | str, sequence: FrameSequence) -> None:
    """
    Saves data from a FrameSequence object to disk.
    """
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    for k, v in asdict(sequence).items():
        with open(path / f'{k}.pt', 'wb') as f:
            torch.save(v, f)


def save_sequence_nerfstudio(path: Path | str, sequence: FrameSequence) -> None:
    """
    Saves data from a FrameSequence object to disk in nerfstudio format.
    """
    pass