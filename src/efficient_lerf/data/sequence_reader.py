import os
import json
from abc import abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE
from nerfstudio.cameras.camera_utils import get_distortion_params

from efficient_lerf.data.sequence import FrameSequence


class FrameSequenceReader:
    """
    """
    def __init__(self, base_dir: Path | str, name: str):
        """
        """
        self.name = name
        self.base_dir = Path(base_dir)
        self.data_dir = Path(base_dir) / name

    def read(self, slice=(0, None, 1), transforms_filename='transforms.json') -> FrameSequence:
        """
        Read sequence from data dir
        """
        transforms_path = self.data_dir / transforms_filename
        if transforms_path.exists():
           transforms = json.load(open(transforms_path, 'r'))
        else:
            transforms = self.load_transforms(transforms_filename)
            json.dump(transforms, open(transforms_path, 'w')) # save transforms to data dir
        sequence = self.read_sequence(slice, transforms)
        sequence.metadata.update({'data_dir': self.data_dir})
        return sequence
    
    @abstractmethod
    def read_sequence(self, slice: tuple, transforms: dict) -> FrameSequence:
        """
        Given nerfstudio transforms, read the sequence.
        """

    @abstractmethod
    def load_transforms(self, filename: Path | str) -> dict:
        """
        Generate nerfstudio transforms from data dir structure.
        """
        pass
    
    @classmethod
    def load_image(cls, filename: Path | str, resize: tuple[int, int]=None):
        """
        Default function to load and resize RGB image.
        """
        image = cv2.imread(str(filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, resize, interpolation=cv2.INTER_LINEAR) if resize else image
        return torch.from_numpy(image)


class LERFFrameSequenceReader(FrameSequenceReader):
    """
    """
    def __init__(self, base_dir: Path | str, name: str, downscale=4):
        """
        """
        super().__init__(base_dir, name)
        self.data_dir = self.base_dir / name / name
        assert downscale in [1, 2, 4, 8], 'Downscale must be 1, 2, 4, or 8'
        self.downscale = downscale

    def read_sequence(self, slice: tuple, transforms: dict) -> FrameSequence:
        """
        """
        def extract_frames(key) -> list:
            if key in transforms:
                return transforms[key]
            frames = [data[key] for data in transforms['frames']]
            frames = frames[slice[0]:slice[1]:slice[2]] if slice[1] else frames[slice[0]::slice[2]]
            return frames
        
        def filename_downscale(filename: str) -> str:
            return filename.replace('images', f'images_{self.downscale}')

        CAMERA_INTRINSICS_DISTORTION = ['k1', 'k2', 'k3', 'k4', 'p1', 'p2']
        distortion_params = {k: transforms[k] for k in CAMERA_INTRINSICS_DISTORTION if k in transforms}
        cameras = Cameras(
            camera_to_worlds=torch.tensor(extract_frames('transform_matrix'))[:, :3, :], # nerfstudio 3x4 convention
            fx=torch.tensor(extract_frames('fl_x')),
            fy=torch.tensor(extract_frames('fl_y')),
            cx=torch.tensor(extract_frames('cx')),
            cy=torch.tensor(extract_frames('cy')),
            width =torch.tensor(extract_frames('w')), 
            height=torch.tensor(extract_frames('h')),
            camera_type=CAMERA_MODEL_TO_TYPE[transforms['camera_model']],
            distortion_params=get_distortion_params(**distortion_params)
        )
        cameras.rescale_output_resolution(scaling_factor=1 / self.downscale)
        image_filenames = [os.path.join(self.data_dir, filename) for filename in extract_frames('file_path')]
        return FrameSequence(
            cameras=cameras, 
            images=torch.stack([self.load_image(filename_downscale(filename)) for filename in image_filenames])
        )

    def load_transforms(self, filename: Path | str) -> dict:
        """
        """
        return json.load(open(self.data_dir / filename, 'r'))
    

if __name__ == '__main__':
    from efficient_lerf.data.common import DATASETS

    for name in DATASETS:
        reader = LERFFrameSequenceReader('/home/gtangg12/data/lerf/LERF Datasets/', name)
        sequence = reader.read(slice=(0, 10, 1))
        print(sequence.cameras.shape)
        print(sequence.images.shape)
        print(sequence.metadata)