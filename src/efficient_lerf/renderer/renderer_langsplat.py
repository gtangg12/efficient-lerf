import os
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
import torchvision

from efficient_lerf.data.common import TorchTensor, DATASET_DIR
from efficient_lerf.renderer.renderer import Renderer

import sys
sys.path.append('third_party/LangSplat')
from autoencoder.model import Autoencoder
from scene.cameras import Camera
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel, render
sys.path.pop()


NUM_SAM_SCALES = 3


def load_pipeline_gaussian(checkpoint: Path | str, feature_scale: int) -> callable:
    """
    """
    model_path = Path(checkpoint) / f'finetune_{feature_scale}'
    with open(model_path / 'cfg_args') as f:
        config = eval(f.read())

    parser = ArgumentParser()
    model, pipeline = ModelParams(parser, sentinel=True), PipelineParams(parser) 
    args = vars(parser.parse_args()) | vars(config)
    args = Namespace(include_feature=True, **args)
    model, pipeline = model.extract(args), pipeline.extract(args)

    gaussian_params, _ = torch.load(model_path / 'chkpnt30000.pth')
    gaussians = GaussianModel(model.sh_degree)
    gaussians.restore(gaussian_params, args, mode='test')
    background = torch.tensor([1, 1, 1]) if model.white_background else \
                 torch.tensor([0, 0, 0])
    background = background.float().to(args.data_device)

    def render_fn(camera: Camera) -> dict:
        return render(camera, gaussians, pipeline, background, args)
    return render_fn


def load_pipeline_autoencoder(checkpoint: Path | str, device='cuda') -> callable:
    """
    """
    encoder_hidden_dims = [256, 128, 64, 32, 3]
    decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]
    checkpoint = torch.load(checkpoint / 'autoencoder/best_ckpt.pth', map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    def decode_fn(features: TorchTensor[..., 'dim']) -> TorchTensor['N', '...']:
        with torch.no_grad():
            return model.decode(features)
    return decode_fn


class LangSplatRenderer(Renderer):
    """
    LangSplat renderer class that wraps around LangSplat pipeline.

    Defines the following rendering methods:
        - render_clip: Returns iterator for clip features at different scales
    
    Defines the following search methods:
        - find_clip: Returns relevancy scores for each positive language query
    """
    def __init__(self, name: str, device='cuda'):
        """
        Loads LangSplat Gaussian renderer and autoencoder decoder.
        """
        super().__init__(DATASET_DIR / 'lerf_ovs' / name, device)
        self.render_fn = {
            i: load_pipeline_gaussian(self.checkpoint, i) for i in range(1, NUM_SAM_SCALES + 1)
        }
        self.decode_fn = load_pipeline_autoencoder(self.checkpoint, device)

    def feature_names(self) -> dict:
        return {'clip': NUM_SAM_SCALES}
    
    def get_camera_transform(self) -> tuple:
        return 1.0, torch.eye(4)

    def render_clip(self, camera: Camera):
        for i in range(1, NUM_SAM_SCALES + 1):
            features_latent = self.render_fn[i](camera)['language_feature_image'].permute(1, 2, 0)
            H, W, _ = features_latent.shape
            features_latent = features_latent.reshape(H * W, -1)
            features_decode = self.decode_fn(features_latent)
            features_decode = features_decode.reshape(H, W, -1)
            yield features_decode

    def find_clip(self, positives: list[str], features: TorchTensor[..., 'dim']) -> TorchTensor['N', '...']:
        pass


if __name__ == '__main__':
    from efficient_lerf.utils.visualization import *
    from efficient_lerf.data.sequence_reader import LangSplatFrameSequenceReader

    tests = Path('/home/gtangg12/efficient-lerf/tests/langsplat')
    os.makedirs(tests, exist_ok=True)

    sequence = LangSplatFrameSequenceReader('figurines').read()

    renderer = LangSplatRenderer('figurines')
    for i, features in enumerate(renderer.render_clip(sequence.cameras[0])):
        print(features.shape)
        visualize_features(features.cpu().numpy()).save(tests / f'clip_{i}.png')
        #torchvision.utils.save_image(features, tests / f'clip_{i}.png')