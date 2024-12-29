from omegaconf import OmegaConf

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, save_sequence_nerfstudio


class VQFeatureFieldEditor:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def extract(self, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W']) -> FrameSequence:
        """
        """
        # save sequence directly to nerfstudio
        sequence = sequence.clone()
        for i in range(len(sequence)):
            sequence.images[i][~masks[i]] = 0
        save_sequence_nerfstudio(sequence)

    def remove(self, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W']) -> FrameSequence:
        """
        """
        masks = ~masks
        sequence = self.extract(sequence, masks)
        save_sequence_nerfstudio(sequence)

    def edit(self, sequence: FrameSequence, masks: TorchTensor['N', 'H', 'W'], prompt: str):
        """
        """
        pass
        # perform colorization on images (YCB)
        # save sequence to nerfstudio