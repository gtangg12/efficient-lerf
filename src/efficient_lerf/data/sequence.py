from __future__ import annotations

import copy
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.math import pad_poses, upsample_feature_map

import sys
sys.path.append('third_party/LangSplat')
from scene.cameras import Camera
sys.path.pop()


@dataclass
class FrameSequence:
    """
    """
    cameras: Cameras | list[Camera]
    images: TorchTensor['N', 'H', 'W', 3]
    depths: TorchTensor['N', 'H', 'W'] = None

    codebook_vectors: dict[str, TorchTensor['n', 'dim']]         = field(default_factory=dict)
    codebook_indices: dict[str, TorchTensor['N', 'M', 'H', 'W']] = field(default_factory=dict) # M is the number of scales
    
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
            if isinstance(v, torch.Tensor):
                sequence[k] = v[indices]
            elif isinstance(v, Cameras):
                sequence[k] = v[indices]
            elif isinstance(v, list):
                sequence[k] = [v[i] for i in indices]
            else:
                sequence[k] = v
        sequence['codebook_vectors'] = {k: v[indices] for k, v in self.codebook_vectors.items()}
        sequence['codebook_indices'] = {k: v[indices] for k, v in self.codebook_indices.items()}
        return FrameSequence(**sequence)
    
    def __repr__(self) -> str:
        """
        """
        images_shape = tuple(self.images.shape) if isinstance(self.images, torch.Tensor) else None
        depths_shape = tuple(self.depths.shape) if isinstance(self.depths, torch.Tensor) else None

        codebook_vectors_shapes = {k: tuple(v.shape) for k, v in self.codebook_vectors.items()}
        codebook_indices_shapes = {k: tuple(v.shape) for k, v in self.codebook_indices.items()}

        return (
            f"FrameSequence(\n"
            f"  cameras={len(self.cameras)},\n"
            f"  images={images_shape},\n"
            f"  depths={depths_shape},\n"
            f"  codebook_vectors={codebook_vectors_shapes},\n"
            f"  codebook_indices={codebook_indices_shapes},\n"
            f"  metadata={self.metadata}\n"
            f")"
        )
    
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
    
    def feature_map(self, name: str, index: int, scale: int = None, upsample=True) -> TorchTensor['H', 'W', 'd']:
        """
        """
        features = self.codebook_vectors[name][self.codebook_indices[name][index][scale]]
        if upsample:
            H = self.cameras[0].height
            W = self.cameras[0].width
            features = upsample_feature_map(features, H, W)
        return features


def load_sequence(path: Path | str) -> FrameSequence:
    """
    Loads data from disk into a FrameSequence object, overwriting existing data.
    """
    return torch.load(path)


def save_sequence(path: Path | str, sequence: FrameSequence) -> None:
    """
    Saves data from a FrameSequence object to disk.
    """
    torch.save(sequence, path)


def save_sequence_nerfstudio(path: Path | str, sequence: FrameSequence) -> None:
    """
    Saves image data from a FrameSequence object to disk in nerfstudio format.
    """
    os.makedirs(path / 'images', exist_ok=True)

    frames = []
    for i in range(len(sequence)):
        filename = path / f'images/image_{i}.png'
        Image.fromarray(sequence.images[i].numpy()).save(filename)
        frames.append({
            'fl_x': sequence.cameras[i].fx,
            'fl_y': sequence.cameras[i].fy,
            'cx': sequence.cameras[i].cx,
            'cy': sequence.cameras[i].cy,
            'w': sequence.cameras[i].width,
            'h': sequence.cameras[i].height,
            'file_path': filename,
            'transform_matrix': pad_poses(sequence.cameras[i].camera_to_worlds[i].numpy())
        })
    return {'camera_model': 'OPENCV', 'frames': frames}