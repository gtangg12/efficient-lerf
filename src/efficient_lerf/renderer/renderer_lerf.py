import yaml
from copy import deepcopy
from pathlib import Path

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from efficient_lerf.data.common import TorchTensor, parent
from efficient_lerf.renderer.renderer import Renderer


def load_pipeline(outputs: Path | str, device='cuda') -> VanillaPipeline:
    """ 
    Load LERF nerfstudio pipeline from checkpoint dir `outputs`.
    """
    with open(Path(outputs) / 'config.yml') as f:
        config = yaml.unsafe_load(f)
        config.load_dir = Path(outputs) / 'nerfstudio_models'
    pipeline = config.pipeline.setup(device=device)
    pipeline.eval()
    eval_load_checkpoint(config, pipeline)
    return pipeline


class RendererLERF(Renderer):
    """
    """
    def __init__(self, load_config: Path | str, device='cuda'):
        """
        Constructs LERF pipeline and rendering scales.
        """
        super().__init__()
        self.device = device
        self.pipeline = load_pipeline(parent(load_config), self.device)
        self.model = self.pipeline.model
        self.model.render_setting = None

        self.scales = torch.linspace(
            0.0,
            self.pipeline.model.config.max_scale,
            self.pipeline.model.config.n_scales
        ).tolist()
        self.scale2index = {scale: i for i, scale in enumerate(self.scales)}
        
        self.disable_model_cache()

    def feature_names(self) -> dict:
        return {'clip': 30, 'dino': 1}
    
    def get_train_cameras(self) -> Cameras:
        return self.pipeline.datamanager.train_dataset.cameras
    
    def get_camera_transform(self) -> tuple:
        return self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale, \
               self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform

    def render_clip(self, camera: Cameras):
        try:
            self.enable_model_cache()
            for scale in self.scales:
                yield self.render(camera, setting=float(scale))['clip']
        finally:
            self.disable_model_cache()

    def render_dino(self, camera: Cameras):
        yield self.render(camera, setting='no_relevancy')['dino']
    
    def render(self, camera: Cameras, setting=None) -> dict:
        """
        Calls renderer at given `camera` with configuration `setting`.
        """
        self.model.render_setting = setting
        outputs = self.model.get_outputs_for_camera(camera.to(self.pipeline.device))
        self.model.render_setting = None # Clear render setting
        return outputs

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
    from tqdm import tqdm

    renderer = RendererLERF('/home/gtangg12/efficient-lerf/outputs/bouquet/lerf/2024-11-07_112933/config.yml')
    cameras = renderer.get_train_cameras()
    cameras.rescale_output_resolution(0.25)
    cameras = cameras[0]
    print(cameras.height, cameras.width)

    import os
    os.makedirs('/home/gtangg12/efficient-lerf/tests/lerf/tensors', exist_ok=True)
    outputs = renderer.render(cameras)
    for k, v in outputs.items():
        print(k, v.shape)
        torch.save(v.cpu(), f'/home/gtangg12/efficient-lerf/tests/lerf/tensors/{k}.pt')
    
    clip = renderer.render(cameras, setting=1.0)['clip'] # identical to outputs['clip'] which has same scale=1
    print(clip.shape)

    from efficient_lerf.utils.visualization import *

    image = (outputs['rgb'].cpu().numpy() * 255).astype('uint8')
    visualize_image(image).save('/home/gtangg12/efficient-lerf/tests/lerf/rgb.png')
    visualize_depth(outputs['depth'].squeeze(2).cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/depth.png')
    visualize_features(outputs['dino'].cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/dino.png')

    pca = compute_pca(clip, n=3, use_torch=True).cpu()
    visualize_features(clip.cpu().numpy(), pca=pca).save('/home/gtangg12/efficient-lerf/tests/lerf/clip_scale.png')
    visualize_features(outputs['clip'].cpu().numpy(), pca=pca).save('/home/gtangg12/efficient-lerf/tests/lerf/clip.png')

    for name in renderer.feature_names():
        for j, embed in tqdm(enumerate(renderer.render_features(name, cameras))):
            visualize_features(embed.cpu().numpy()).save(f'/home/gtangg12/efficient-lerf/tests/lerf/{name}_{j}.png')