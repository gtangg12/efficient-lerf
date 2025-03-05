import os

import cv2
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, save_sequence_nerfstudio
from efficient_lerf.models.model_instructpix2pix import ModelInstructPix2Pix
from efficient_lerf.utils.math import resize_feature_map
from efficient_lerf.utils.visualization import *


class VQFeatureFieldEditor:
    """
    """
    def __init__(self):
        """
        """
        self.edit_model = ModelInstructPix2Pix()

    def extract(self, name: str, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W']) -> FrameSequence:
        """
        """
        # save sequence directly to nerfstudio
        sequence = sequence.clone()
        for i in range(len(sequence)):
            sequence.images[i][~masks[i]] = 0
        save_sequence_nerfstudio(sequence.metadata['data_dir'] / f'sequence/nerfstudio/extract_{name}', sequence)
        return sequence

    def remove(self, name: str, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W']) -> FrameSequence:
        """
        """
        masks = ~masks
        sequence = self.extract(sequence, masks)
        save_sequence_nerfstudio(sequence.metadata['data_dir'] / f'sequence/nerfstudio/remove_{name}', sequence)
        return sequence
        
    def edit(self, name: str, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W'], prompt: str, save=True, debug=False) -> FrameSequence:
        """
        """
        if debug:
            os.makedirs(f"tests/editing/{name}")

        sequence, masks = sequence.clone(), masks.clone()
        _, H, W, _ = sequence.images.shape

        masks_resized = []
        for i, mask in enumerate(tqdm(masks)):
            m = resize_feature_map(mask.unsqueeze(-1), H, W).squeeze(-1)
            if debug:
                visualize_relevancy(m.float().numpy()).save(f"tests/editing/{name}/mask_{i:03}.png")
            masks_resized.append(m)
        masks_resized = torch.stack(masks_resized)
        
        edited_images = []
        for i, (image, mask) in enumerate(tqdm(zip(sequence.images, masks_resized))):
            edited = self.edit_model(prompt, image, downsample=2)
            if debug:
                visualize_image(edited.detach().cpu().numpy()).save(f"tests/editing/{name}/edited_{i:03}.png")
            edited = image * (~mask[..., None]) + edited * mask[..., None]
            edited_images.append(edited)
            if debug:
                visualize_image(edited.detach().cpu().numpy()).save(f"tests/editing/{name}/edited_masked_{i:03}.png")
        
        sequence.images = torch.stack(edited_images)
        if save:
            save_sequence_nerfstudio(sequence.metadata['data_dir'] / f'sequence/nerfstudio/edit_{name}', sequence)
        return sequence