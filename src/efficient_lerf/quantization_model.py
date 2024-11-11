import torch
from omegaconf import OmegaConf
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.sequence import FrameSequence, sequence_to_point_cloud
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.quantization import *


class DiscreteFeatureField:
    """
    """
    def __init__(self, config: OmegaConf, sequence: FrameSequence):
        """
        """
        self.config = config
        self.renderer = Renderer(self.config.renderer)
        self.sequence = self.quantize(sequence, self.renderer)
        self.sequence_point_cloud = sequence_to_point_cloud(self.sequence)
        self.scale2index = {scale: i for i, scale in enumerate(self.renderer.scales)}
    
    def quantize(self, sequence, renderer) -> FrameSequence:
        """
        """
        quant_cam_traj = CameraTrajQuantization(self.config.quantization.camera_traj)
        quant_feat_map = FeatureMapQuantization(self.config.quantization.feature_map)
        sequence = quant_cam_traj.process_sequence(sequence)
        sequence = quant_feat_map.process_sequence(sequence, renderer)
        return sequence

    def render(self, camera: Cameras, scale: float) -> dict:
        """
        """
        outputs = self.sequence_point_cloud.render(camera)
        clip_codebook = self.sequence.clip_codebook
        dino_codebook = self.sequence.dino_codebook
        clip_codebook_indices = outputs['clip_codebook_indices']
        dino_codebook_indices = outputs['dino_codebook_indices']
        y = outputs['coords'][:, 0]
        x = outputs['coords'][:, 1]
        clip_embeds = torch.zeros(camera.height, camera.width, clip_codebook[..., self.scale2index(scale)].shape[-1])
        dino_embeds = torch.zeros(camera.height, camera.width, dino_codebook.shape[-1])
        clip_embeds[y, x] = clip_codebook[clip_codebook_indices]
        dino_embeds[y, x] = dino_codebook[dino_codebook_indices]
        return {
            'clip': clip_embeds,
            'dino': dino_embeds,
        }