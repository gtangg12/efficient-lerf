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
    def __init__(self, config: OmegaConf, sequence: FrameSequence, renderer: Renderer, compute_pc=True):
        """
        """
        self.config = config
        self.renderer = renderer
        self.sequence = self.quantize(sequence, self.renderer)
        if compute_pc:
            self.sequence_point_cloud = sequence_to_point_cloud(self.sequence)
        # print(self.sequence_point_cloud.points.shape)
        # print(self.sequence_point_cloud.depths.shape)
        # print(self.sequence_point_cloud.clip_codebook_indices.shape)
        # print(self.sequence_point_cloud.dino_codebook_indices.shape)
    
    def quantize(self, sequence, renderer) -> FrameSequence:
        """
        """
        path = sequence.metadata['data_dir'] / 'sequence'
        #if path.exists():
        #    return load_sequence(path)

        quant_cam_traj = CameraTrajQuantization(self.config.camera_traj_quant)
        quant_feat_map = FeatureMapQuantization(self.config.feature_map_quant)
        sequence, indices = quant_cam_traj.process_sequence(sequence)
        print(indices, len(sequence))
        sequence = quant_feat_map.process_sequence(sequence, renderer)

        save_sequence(path, sequence)
        return sequence
    
    def exist(self, positives: list[str], return_pc=False, threshold=0.5) -> FrameSequencePointCloud:
        """
        """
        image_encoder = self.renderer.pipeline.image_encoder
        image_encoder.set_positives(positives)
        
        codebook_mask = torch.zeros(self.sequence.clip_codebook.shape[0], dtype=torch.bool)
        scores = defaultdict(lambda: 0)

        for i in range(len(positives)):
            positive = positives[i]
            codebook = self.sequence.clip_codebook.to(self.renderer.device)
            probs = image_encoder.get_relevancy(codebook, positive_id=i)
            codebook_mask |= probs[:, 0].cpu() > threshold
            scores[positive] = probs[:, 0].max().item() # positive prob

        if return_pc:
            return scores, self.sequence_point_cloud.subsample(codebook_mask, feature='clip')
        return scores
    
    def render(self, positives: list[str], camera: Cameras, threshold=0.5, return_relevancy=False) -> dict:
        """
        """
        _, point_cloud = self.exist(positives, return_pc=True, threshold=threshold)
        outputs = point_cloud.render_features(camera)
        if not return_relevancy:
            return outputs
        
        assert len(positives) == 1, 'Only one positive is supported for relevancy visualization'

        valid = outputs['valid_mask']
        score = torch.zeros(camera.height, camera.width)
        for k, features in outputs.items():
            if k in ['valid_mask', 'dino']:
                continue
            features = features.flatten(0, -2).to(self.renderer.device)
            probs = self.renderer.pipeline.image_encoder.get_relevancy(features, positive_id=0)
            probs = probs[:, 0].cpu().reshape(camera.height, camera.width)
            score = torch.maximum(score, probs)
        score[~valid] = 0
        return {'relevancy': score, **outputs}


def load_model(scene: str, config: OmegaConf, **kwargs) -> DiscreteFeatureField:
    """
    """    
    sequence_reader = LERFFrameSequenceReader(DATASET_DIR, scene)
    sequence = sequence_reader.read(slice=tuple(config.slice))
    renderer = Renderer(load_checkpoint(scene))
    config.feature_map_quant.visualize_dir = sequence_reader.data_dir / 'sequence/visualizations' if config.visualize else None
    config.feature_map_quant.visualize_iter = config.visualize_iter 
    return DiscreteFeatureField(config, sequence, renderer, **kwargs)


if __name__ == '__main__':
    from efficient_lerf.data.common import DATASETS

    for scene in DATASETS:
        model = load_model(scene, OmegaConf.load(CONFIGS_DIR / 'template.yaml'), compute_pc=False)