from glob import glob

import torch
from omegaconf import OmegaConf
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.common import *
from efficient_lerf.data.sequence import *
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.quantization import *


class DiscreteFeatureField:
    """
    """
    def __init__(self, config: OmegaConf, sequence: FrameSequence, renderer: Renderer):
        """
        """
        self.config = config
        self.renderer = renderer
        self.sequence = self.quantize(sequence, self.renderer)
        self.sequence_point_cloud = sequence_to_point_cloud(self.sequence)
        # print(self.sequence_point_cloud.points.shape)
        # print(self.sequence_point_cloud.depths.shape)
        # print(self.sequence_point_cloud.clip_codebook_indices.shape)
        # print(self.sequence_point_cloud.dino_codebook_indices.shape)
    
    def quantize(self, sequence, renderer) -> FrameSequence:
        """
        """
        path = sequence.metadata['data_dir'] / 'sequence'
        if path.exists():
            return load_sequence(path)

        quant_cam_traj = CameraTrajQuantization(self.config.camera_traj_quant)
        quant_feat_map = FeatureMapQuantization(self.config.feature_map_quant)
        sequence = quant_cam_traj.process_sequence(sequence)
        sequence = quant_feat_map.process_sequence(sequence, renderer)

        save_sequence(path, sequence)
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
        clip_embeds = torch.zeros(camera.height, camera.width, clip_codebook[..., self.renderer.scale2index(scale)].shape[-1])
        dino_embeds = torch.zeros(camera.height, camera.width, dino_codebook.shape[-1])
        clip_embeds[y, x] = clip_codebook[clip_codebook_indices]
        dino_embeds[y, x] = dino_codebook[dino_codebook_indices]
        return {
            'clip': clip_embeds,
            'dino': dino_embeds,
        }


def load_model(scene: str, config: OmegaConf) -> DiscreteFeatureField:
    """
    """
    def load_checkpoint():
        return glob(f'{OUTPUTS_DIR}/{scene}/lerf/*/config.yml')[0]
    
    sequence_reader = LERFFrameSequenceReader(DATASET_DIR, scene)
    sequence = sequence_reader.read(slice=tuple(config.slice))
    renderer = Renderer(load_checkpoint())
    config.visualize_dir = sequence_reader.data_dir / 'sequence/visualizations' if config.visualize else None
    return DiscreteFeatureField(config, sequence, renderer)


if __name__ == '__main__':
    model = load_model('bouquet', OmegaConf.load(CONFIGS_DIR / 'template.yaml'))

    from efficient_lerf.utils.visualization import *