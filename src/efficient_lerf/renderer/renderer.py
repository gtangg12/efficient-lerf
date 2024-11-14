import yaml
from copy import deepcopy
from pathlib import Path

import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint

from efficient_lerf.data.common import TorchTensor, parent


def load_pipeline(outputs: Path | str, device='cuda') -> VanillaPipeline:
    """
    """
    with open(Path(outputs) / 'config.yml') as f:
        config = yaml.unsafe_load(f)
        config.load_dir = Path(outputs) / 'nerfstudio_models'
    pipeline = config.pipeline.setup(device=device)
    pipeline.eval()
    eval_load_checkpoint(config, pipeline)
    return pipeline


class Renderer:
    """
    """
    def __init__(self, load_config: Path | str, device='cuda'):
        """
        """
        self.load_config = load_config
        self.device = device
        self.load_pipeline()
        self.scales = torch.linspace(
            0.0,
            self.pipeline.model.config.max_scale,
            self.pipeline.model.config.n_scales
        ).tolist()
        self.scale2index = {scale: i for i, scale in enumerate(self.scales)}
        
        self.disable_model_cache()

    def load_pipeline(self):
        """
        """
        self.pipeline = load_pipeline(parent(self.load_config), self.device)
        self.model = self.pipeline.model
        self.model.render_setting = None

    def unload_pipeline(self):
        """
        """
        del self.pipeline
        del self.model

    def render_setting(self, camera: Cameras, setting=None) -> dict:
        """
        """
        assert camera.camera_to_worlds.ndim == 2, 'Only one camera is supported'

        self.model.render_setting = setting
        outputs = self.model.get_outputs_for_camera(camera.to(self.pipeline.device))
        self.model.render_setting = None # Clear render setting
        return outputs

    def render(self, camera: Cameras) -> dict:
        """
        """
        return self.render_setting(camera, setting='base')

    def render_scale(self, camera: Cameras, scale: float) -> TorchTensor['H', 'W', 'clip_dim']:
        """
        """
        return self.render_setting(camera, setting=float(scale))['clip']
    
    def render_vanilla(self, camera: Cameras) -> dict:
        """
        """
        return self.render_setting(camera, setting=None)
    
    def get_train_cameras(self) -> Cameras:
        """
        """
        return self.pipeline.datamanager.train_dataset.cameras
    
    def get_camera_transform(self) -> tuple:
        """
        """
        return self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale, \
               self.pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_transform
    
    def disable_model_cache(self):
        """
        """
        self.model.use_cache = False
        self.model.cache = {}

    def enable_model_cache(self):
        """
        """
        self.model.use_cache = True
        self.model.cache = {}


if __name__ == '__main__':
    renderer = Renderer('/home/gtangg12/efficient-lerf/outputs/bouquet/lerf/2024-11-07_112933/config.yml')
    cameras = renderer.get_train_cameras()[0]

    import os
    os.makedirs('/home/gtangg12/efficient-lerf/tests/lerf/tensors', exist_ok=True)
    outputs = renderer.render(cameras)
    for k, v in outputs.items():
        print(k, v.shape)
        torch.save(v.cpu(), f'/home/gtangg12/efficient-lerf/tests/lerf/tensors/{k}.pt')
    
    clip = renderer.render_scale(cameras, 1) # identical to outputs['clip'] which has same scale=1
    print(clip.shape)

    from efficient_lerf.utils.visualization import *

    image = (outputs['rgb'].cpu().numpy() * 255).astype('uint8')
    visualize_image(image).save('/home/gtangg12/efficient-lerf/tests/lerf/rgb.png')
    visualize_depth(outputs['depth'].squeeze(2).cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/depth.png')
    visualize_features(outputs['clip'].cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/clip.png')
    visualize_features(outputs['dino'].cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/dino.png')
    visualize_features(clip.cpu().numpy()).save('/home/gtangg12/efficient-lerf/tests/lerf/clip_scale.png')

    from tqdm import tqdm
    for scale in tqdm(renderer.scales):
        clip = renderer.render_scale(cameras, scale)
        visualize_features(clip.cpu().numpy()).save(f'/home/gtangg12/efficient-lerf/tests/lerf/clip_scale_{scale}.png')