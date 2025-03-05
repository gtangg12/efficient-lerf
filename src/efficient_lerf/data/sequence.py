from __future__ import annotations

import copy
import json
import os
from argparse import Namespace
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.math import norm, pad_poses, resize_feature_map

import sys
sys.path.append('third_party/LangSplat')
from scene.cameras import Camera
sys.path.pop()


def rescale_tensor(tensor: TorchTensor['B', 'H', 'W', 'C'], scale: float, mode='bilinear') -> torch.Tensor:
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor.permute(0, 2, 3, 1)


def gsplat_camera2namespace(camera: Camera) -> Namespace:
    """
    Serialize a gsplat Camera object to a Namespace object for parameter passing.
    """
    return Namespace(
        colmap_id=camera.colmap_id,
        R=camera.R,
        T=camera.T,
        FoVx=camera.FoVx,
        FoVy=camera.FoVy,
        image=camera.original_image,
        image_name=camera.image_name,
        uid=camera.uid,
        gt_alpha_mask=None,
    )


@dataclass
class FrameSequence:
    """
    """
    cameras: Cameras | list[Camera] # support nerfstudio and gsplat cameras
    images: TorchTensor['N', 'H', 'W', 3]
    depths: TorchTensor['N', 'H', 'W'] = None

    codebook_vectors: dict[str, TorchTensor['n', 'dim']]         = field(default_factory=dict)
    codebook_indices: dict[str, TorchTensor['N', 'M', 'H', 'W']] = field(default_factory=dict) # M is the number of scales
    
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        """
        """
        return len(self.images)
    
    def __getitem__(self, indices: slice) -> FrameSequence:
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
                sequence[k] = v[indices]
            else:
                sequence[k] = v
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
    
    def rescale_camera_resolution(self, scale: float) -> None:
        """
        """
        if isinstance(self.cameras, Cameras):
            self.cameras.rescale_output_resolution(scaling_factor=scale)
        else:
            outputs = []
            for camera in self.cameras:
                namespace = gsplat_camera2namespace(camera) # gsplat does not scale fx, fy, cx, cy
                namespace.image = F.interpolate(camera.original_image[None, ...], scale_factor=scale, mode='bilinear')[0]
                outputs.append(Camera(**vars(namespace)))
            self.cameras = outputs
        
        self.images = rescale_tensor(self.images, scale)
        self.depths = rescale_tensor(self.depths, scale, mode='nearest') if self.depths is not None else None
        # TODO: rescale codebook_indices

    def transform_cameras(self, scale: float, trans: TorchTensor[4, 4]) -> None:
        """
        """
        if isinstance(self.cameras, Cameras):
            self.cameras.camera_to_worlds = trans @ pad_poses(self.cameras.camera_to_worlds)
            self.cameras.camera_to_worlds[:, :3, 3] *= scale
        else:
            outputs = []
            for camera in self.cameras:
                namespace = gsplat_camera2namespace(camera)
                namespace.R = (trans[:3, :3] @ camera.R).numpy()
                namespace.T = ((trans[:3, 3] + camera.T) * scale).numpy()
                outputs.append(Camera(**vars(namespace)))
            self.cameras = outputs

    def feature_map(self, name: str, index: int, scale: int = None, upsample=True) -> TorchTensor['H', 'W', 'd']:
        """
        """
        features = self.codebook_vectors[name][self.codebook_indices[name][index][scale]]
        if upsample:
            H = self.images[0].shape[0]
            W = self.images[0].shape[1]
            features = resize_feature_map(features, H, W)
        features = norm(features, dim=-1)
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
            'fl_x': sequence.cameras[i].fx.item(),
            'fl_y': sequence.cameras[i].fy.item(),
            'cx': sequence.cameras[i].cx.item(),
            'cy': sequence.cameras[i].cy.item(),
            'w': sequence.cameras[i].width.item(),
            'h': sequence.cameras[i].height.item(),
            'file_path': str(filename),
            'transform_matrix': pad_poses(sequence.cameras.camera_to_worlds[i]).numpy().tolist()
        })
    transforms = {'camera_model': 'OPENCV', 'frames': frames}
    with open(path / 'transforms.json', 'w') as f:
        json.dump(transforms, f)