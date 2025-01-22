import torch
from omegaconf import OmegaConf

from efficient_lerf.data.common import *
from efficient_lerf.data.sequence import *
from efficient_lerf.data.sequence_reader import *
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.renderer.renderer_lerf import LERFRenderer
from efficient_lerf.renderer.renderer_langsplat import LangSplatRenderer
from efficient_lerf.quantization import *
from efficient_lerf.quantization_methods import *
from efficient_lerf.feature_field_editor import VQFeatureFieldEditor
from efficient_lerf.utils.math import compute_relevancy


class VQFeatureField:
    """
    """
    def __init__(self, config: OmegaConf, sequence: FrameSequence, renderer: Renderer, device='cuda'):
        """
        """
        self.config = config
        self.device = device
        self.renderer = renderer
        self.sequence = self.quantize(sequence)
        self.sequence_editor = VQFeatureFieldEditor()
    
    def quantize(self, sequence: FrameSequence) -> FrameSequence:
        """
        """
        self.feature_map_quant = FeatureMapQuantization(self.config.feature_map_quant)

        path = sequence.metadata['data_dir'] / 'sequence/sequence.pt'
        if path.exists():
            return load_sequence(path)
        sequence = self.feature_map_quant.process_sequence(sequence, self.renderer)
        save_sequence(path, sequence)
        return sequence
    
    def find(self, name: str, positives: any, threshold=0.5) -> tuple[
        TorchTensor['N'],
        TorchTensor['N', 'H', 'W']
    ]:
        """
        """
        codebook_vectors = self.sequence.codebook_vectors[name].to(self.device) # (k, dim)
        codebook_indices = self.sequence.codebook_indices[name].to(self.device) # (N, M, H, W)

        probs = self.renderer.find(name, positives, codebook_vectors) # (N, k)
        
        scores, relevancy_maps = [], []
        for i, prob in enumerate(probs):
            scores.append(prob.max().item())
            relevancy_maps.append(compute_relevancy(prob[codebook_indices], threshold).cpu())
        return torch.tensor(scores), torch.stack(relevancy_maps)

    def edit(self, edit_method='extract', find_name='clip', positive=None, threshold=None, **kwargs) -> None:
        """
        """
        assert len(positive) == 1, 'Only one positive query per edit is supported'
        assert edit_method in ['extract', 'remove', 'edit']
        _, relevancy_maps = self.find(find_name, positive, threshold)

        getattr(self.sequence_editor, edit_method)(self.sequence, relevancy_maps, **kwargs)


def load_model(scene: str, config: OmegaConf, RendererT: type, FrameSequenceReaderT: type) -> VQFeatureField:
    """
    """
    reader = FrameSequenceReaderT(scene)
    sequence = reader.read(slice=tuple(config.slice))
    renderer = RendererT(scene)
    config.feature_map_quant.visualize_dir = reader.data_dir / 'sequence/visualizations'
    return VQFeatureField(config, sequence, renderer)


if __name__ == '__main__':
    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

    model = load_model('figurines', config, LERFRenderer, LERFFrameSequenceReader)
    scores, relevancy_maps = model.find('clip', ['jake from adventure time', 'toy', 'apple pie'])
    print(scores)
    print(relevancy_maps.shape)