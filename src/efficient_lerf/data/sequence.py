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
    clip_codebook_indices: TorchTensor['N', 'nscales', 'H', 'W'] = None
    dino_codebook_indices: TorchTensor['N', 'H', 'W'] = None
    
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.images)
    
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


@dataclass
class FrameSequencePointCloud:
    """
    """
    points: TorchTensor['N', 3]
    depths: TorchTensor['N']
    clip_codebook_indices: TorchTensor['N', 'M']
    dino_codebook_indices: TorchTensor['N']

    def __len__(self):
        """
        """
        return len(self.points)
    
    def render(self, camera: Cameras) -> dict:
        """
        """
        assert len(camera) == 1, 'Only one camera supported'

        M = self.clip_codebook_indices.shape[-1]

        points = torch.inverse(camera.camera_to_worlds) @ self.points.T
        x_proj = points[:, 0] / points[:, 2]
        y_proj = points[:, 1] / points[:, 2]
        depths = points[:, 2]
        u = camera.fx * x_proj + camera.cx
        v = camera.fy * y_proj + camera.cy

        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depths > 0)
        u = u[valid]
        v = v[valid]
        
        coords = torch.stack([u, v], dim=1).long()
        depths = depths[valid]
        depths, indices = torch.sort(depths)
        coords = coords[indices]
        depths = depths[indices]
        vindex = torch.where(valid)[indices]

        clip_codebook_indices = torch.zeros(camera.height, camera.width, M, dtype=torch.long)
        dino_codebook_indices = torch.zeros(camera.height, camera.width   , dtype=torch.long)
        clip_codebook_indices[coords[:, 1], coords[:, 0]] = self.clip_codebook_indices[vindex]
        dino_codebook_indices[coords[:, 1], coords[:, 0]] = self.dino_codebook_indices[vindex]
        return {
            'coords': coords,
            'clip_codebook_indices': self.clip_codebook_indices,
            'dino_codebook_indices': self.dino_codebook_indices,
        }


def sequence_to_point_cloud(sequence: FrameSequence) -> FrameSequencePointCloud:
    """
    """
    depths = sequence.depths.flatten()[:, None]
    bundle = sequence.cameras.generate_rays(torch.arange(len(sequence.cameras)))
    points = bundle.origins + bundle.directions * depths
    valid = depths != 0 # remove background points
    return FrameSequencePointCloud(
        points=points[valid], 
        depths=depths[valid],
        sequence=sequence,
    )