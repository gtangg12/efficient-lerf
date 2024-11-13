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
    colors: TorchTensor['N', 3]
    points: TorchTensor['N', 3]
    depths: TorchTensor['N']
    clip_codebook_indices: TorchTensor['N', 'M']
    dino_codebook_indices: TorchTensor['N']
    sequence: FrameSequence # reference to the original sequence for codebooks

    def __len__(self):
        """
        """
        return len(self.points)
    
    def subsample(self, codebook_mask: TorchTensor['n_dim'], feature='clip') -> FrameSequencePointCloud:
        """
        """
        assert feature in ['clip', 'dino']
        mask = codebook_mask[self.clip_codebook_indices] if feature == 'clip' else \
               codebook_mask[self.dino_codebook_indices]
        if feature == 'clip':
            mask = mask.any(dim=1)      
        return FrameSequencePointCloud(
            colors=self.colors[mask],
            points=self.points[mask],
            depths=self.depths[mask],
            clip_codebook_indices=self.clip_codebook_indices[mask],
            dino_codebook_indices=self.dino_codebook_indices[mask],
            sequence=self.sequence,
        )
    
    def render_indices(self, camera: Cameras) -> dict:
        """
        """
        assert camera.camera_to_worlds.ndim == 2, 'Only one camera supported'

        M = self.clip_codebook_indices.shape[-1]

        camera_to_world = pad_poses(camera.camera_to_worlds)
        points = torch.cat([self.points, torch.ones(len(self.points), 1)], dim=1)
        points = torch.inverse(camera_to_world)[:3, ] @ points.T
        points = points.T
        
        x_proj = -points[:, 0] / points[:, 2]
        y_proj =  points[:, 1] / points[:, 2]
        depths = -points[:, 2]
        u = camera.fx * x_proj + camera.cx
        v = camera.fy * y_proj + camera.cy
        valid = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height) & (depths > 0)
        coords = torch.stack([u, v], dim=1).int()[valid]
        depths = depths[valid]

        depths, indices = torch.sort(depths, descending=True)
        coords = coords[indices]
        depths = depths[indices]
        valid_indices = torch.nonzero(valid).squeeze(1)[indices]

        clip_codebook_indices = torch.full((camera.height, camera.width, M), -1, dtype=torch.int)
        dino_codebook_indices = torch.full((camera.height, camera.width   ), -1, dtype=torch.int)
        clip_codebook_indices[coords[:, 1], coords[:, 0]] = self.clip_codebook_indices[valid_indices]
        dino_codebook_indices[coords[:, 1], coords[:, 0]] = self.dino_codebook_indices[valid_indices]
        return {
            'coords': coords,                                                # (N, 2)
            'clip_codebook_indices': clip_codebook_indices.permute(2, 0, 1), # (H, W, dim_clip)
            'dino_codebook_indices': dino_codebook_indices,                  # (H, W, dim_dino)
        }

    def render_features(self, camera: Cameras) -> dict:
        """
        """
        outputs = self.render_indices(camera)
        coords = outputs['coords']
        clip_codebook_indices = outputs['clip_codebook_indices']
        dino_codebook_indices = outputs['dino_codebook_indices']
        
        def extract_features(codebook, indices):
            features = torch.zeros(*indices.shape, codebook.shape[-1])
            features[coords[:, 1], coords[:, 0]] = codebook[indices[coords[:, 1], coords[:, 0]]]
            return features
        
        valid = torch.zeros(camera.height, camera.width, dtype=torch.bool)
        valid[coords[:, 1], coords[:, 0]] = True

        outputs = {'valid_mask': valid}
        for i, indices in enumerate(clip_codebook_indices):
            outputs[f'clip_{i}'] = extract_features(self.sequence.clip_codebook, indices)
        outputs['dino'] = extract_features(self.sequence.dino_codebook, dino_codebook_indices)
        return outputs


def sequence_to_point_cloud(sequence: FrameSequence) -> TorchTensor['n', 3]:
    """
    """
    def camera_reshape(x):
        return x.permute(2, 0, 1, 3)

    colors = sequence.images.reshape(-1, 3)
    depths = sequence.depths
    bundle = sequence.cameras.generate_rays(torch.arange(len(sequence.cameras))[:, None])
    points = camera_reshape(bundle.origins) + camera_reshape(bundle.directions) * depths[..., None]
    points = points.reshape(-1, 3)
    depths = depths.flatten()
    valid = (depths != 0) & (depths < torch.quantile(depths, 0.99)) # remove background/outlier points
    return FrameSequencePointCloud(
        colors=colors[valid],
        points=points[valid],
        depths=depths[valid],
        clip_codebook_indices=sequence.clip_codebook_indices.permute(0, 2, 3, 1).flatten(0, -2)[valid],
        dino_codebook_indices=sequence.dino_codebook_indices.flatten()[valid],
        sequence=sequence,
    )