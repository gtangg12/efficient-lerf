import cv2
import numpy as np
from omegaconf import OmegaConf

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, save_sequence_nerfstudio
from efficient_lerf.models.model_instructpix2pix import ModelInstructPix2Pix


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
        
    def edit(self, name: str, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W'], prompt: str, dilation=10) -> FrameSequence:
        """
        """
        sequence, masks = sequence.clone(), masks.clone()
        for i, mask in enumerate(masks):
            masks[i] = cv2.dilate(mask, kernel=np.ones((dilation, dilation)))
        sequence.images = self.edit_model(prompt, sequence.images, masks)
        save_sequence_nerfstudio(sequence.metadata['data_dir'] / f'sequence/nerfstudio/edit_{name}', sequence)
        return sequence