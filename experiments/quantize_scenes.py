from omegaconf import OmegaConf

from efficient_lerf.data.common import CONFIGS_DIR
from efficient_lerf.feature_field import load_model

from experiments.common import DATASETS, RENDERERS


if __name__ == '__main__':
    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

    for RendererT, FrameSequenceReaderT in RENDERERS:
        for scene in DATASETS:
            model = load_model(scene, config, RendererT, FrameSequenceReaderT) # auto quantize