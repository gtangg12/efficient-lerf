import yaml
from copy import deepcopy
from pathlib import Path

import torch
from natsort import natsorted
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from efficient_lerf.data.common import TorchTensor, DATASET_DIR
from efficient_lerf.renderer.renderer import Renderer


def load_pipeline(checkpoint: Path, device='cuda') -> VanillaPipeline:
    """ 
    Load LERF nerfstudio pipeline from checkpoint dir `outputs`.
    """
    with open(checkpoint / 'config.yml') as f:
        config = yaml.unsafe_load(f)
        config.load_dir = checkpoint / 'nerfstudio_models'
    pipeline = config.pipeline.setup(device=device)
    pipeline.eval()
    eval_load_checkpoint(config, pipeline)
    return pipeline


def latest_checkpoint(path: Path | str) -> Path:
    path = Path(path) / 'lerf'
    name = natsorted(path.glob('*'))[-1]
    return path / name


class LERFRenderer(Renderer):
    """
    """
    def __init__(self, name: str, device='cuda'):
        """
        Constructs LERF pipeline and rendering scales.
        """
        super().__init__(latest_checkpoint(DATASET_DIR / 'lerf/outputs' / name), device)
        self.pipeline = load_pipeline(self.checkpoint, self.device)
        self.model = self.pipeline.model
        self.model.render_setting = None
        self.train_dataset = self.pipeline.datamanager.train_dataset

        self.scales = torch.linspace(
            0.0,
            self.pipeline.model.config.max_scale,
            self.pipeline.model.config.n_scales
        ).tolist()
        self.scale2index = {scale: i for i, scale in enumerate(self.scales)}
        
        self.disable_model_cache()

    def feature_names(self) -> dict:
        return {'clip': 30, 'dino': 1}
    
    def get_camera_transform(self) -> tuple:
        return self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale, \
               self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

    def render_clip(self, camera: Cameras):
        try:
            self.enable_model_cache()
            for scale in self.scales:
                yield self.render_helper(camera, setting=float(scale))['clip']
        finally:
            self.disable_model_cache()

    def render_dino(self, camera: Cameras):
        yield self.render_helper(camera, setting='no_relevancy')['dino']
    
    def render_lerf(self, camera: Cameras):
        return self.render_helper(camera, setting='no_relevancy')

    def render_helper(self, camera: Cameras, setting=None) -> dict:
        """
        Calls renderer at given `camera` with configuration `setting`.
        """
        self.model.render_setting = setting
        outputs = self.model.get_outputs_for_camera(camera.to(self.pipeline.device))
        self.model.render_setting = None # Clear render setting
        return outputs
    
    def find_clip(self, positives: list[str], features: TorchTensor[..., 'dim']) -> TorchTensor['N', '...']:
        image_encoder = self.pipeline.image_encoder
        image_encoder.set_positives(positives)
        probs = []
        shape, dim = features.shape[:-1], features.shape[-1]
        features = features.view(-1, dim).to(self.device)
        for i in range(len(positives)):
            probs.append(image_encoder.get_relevancy(features, positive_id=i)[:, 0].view(*shape)) # positive prob
        return torch.stack(probs)

    def find_dino(self, positives: TorchTensor['N', 'dim'], features: TorchTensor[..., 'dim']) -> TorchTensor['N', '...']:
        shape, dim = features.shape[:-1], features.shape[-1]
        probs = positives @ features.view(-1, dim).T
        return probs.view(len(positives), *shape)

    def enable_model_cache(self):
        """
        We modify LERF to cache temporary ray bundle data on GPU memory to avoid recomputation over different 
        rendering scales (see internal/lerf.py).
        """
        self.model.use_cache = True
        self.model.cache = {}

    def disable_model_cache(self):
        """
        Reset model cache.
        """
        self.model.use_cache = False
        self.model.cache = {}


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    from efficient_lerf.data.common import DATASET_DIR
    from efficient_lerf.utils.visualization import *

    tests = Path('/home/gtangg12/efficient-lerf/tests/lerf')
    
    renderer = LERFRenderer('bouquet')
    cameras = renderer.pipeline.datamanager.train_dataset.cameras
    cameras.rescale_output_resolution(0.25)
    cameras = cameras[0]
    print(cameras.height, cameras.width)

    os.makedirs(f'{tests}/lerf/tensors', exist_ok=True)
    outputs = renderer.render_lerf(cameras)
    for k, v in outputs.items():
        print(k, v.shape)
        torch.save(v.cpu(), f'{tests}/tensors/{k}.pt')
    
    clip = renderer.render_helper(cameras, setting=1.0)['clip'] # identical to outputs['clip'] which has same scale=1
    dino = outputs['dino']

    positives = ['flower', 'bouquet', 'rose']
    probs = renderer.find_clip(positives, clip)

    for i, positive in enumerate(positives):
        visualize_relevancy(probs[i].cpu().numpy()).save(f'{tests}/relevancy_{positive}.png')

    image = (outputs['rgb'].cpu().numpy() * 255).astype('uint8')
    visualize_image(image).save(f'{tests}/rgb.png')
    visualize_depth(outputs['depth'].squeeze(2).cpu().numpy()).save(f'{tests}/depth.png')
    visualize_features(dino.cpu().numpy()).save(f'{tests}/dino.png')

    pca = compute_pca(clip, n=3, use_torch=True).cpu()
    visualize_features(clip.cpu().numpy(), pca=pca).save(f'{tests}/clip_scale.png')
    visualize_features(outputs['clip'].cpu().numpy(), pca=pca).save(f'{tests}/clip.png')

    for name in renderer.feature_names():
        for j, embed in tqdm(enumerate(renderer.render(name, cameras))):
            visualize_features(embed.cpu().numpy()).save(f'{tests}/{name}_{j}.png')