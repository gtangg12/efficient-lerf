import copy
import json
from glob import glob
from pathlib import Path

import kornia
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from natsort import natsorted
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length

from efficient_lerf.data.common import TorchTensor, DATASET_DIR, CONFIGS_DIR
from efficient_lerf.quantization_model import DiscreteFeatureField, load_model
from efficient_lerf.utils.math import pad_poses
from efficient_lerf.utils.visualization import *


def flatten_bbox(bbox) -> list: # xyxy -> TLBR format
    return [int(bbox[0][1]), int(bbox[0][0]), int(bbox[1][1]), int(bbox[1][0])]


def transform_cameras_inverse(cameras: Cameras, scale: float, trans: TorchTensor[4, 4]) -> Cameras:
    """
    """
    scale = 1 / scale
    trans = torch.inverse(pad_poses(trans))
    cameras = copy.deepcopy(cameras)
    cameras.camera_to_worlds = pad_poses(cameras.camera_to_worlds)
    cameras.camera_to_worlds[..., :3, 3] *= scale # note order
    cameras.camera_to_worlds = trans @ cameras.camera_to_worlds
    cameras.camera_to_worlds = cameras.camera_to_worlds[..., :3, :]
    return cameras


def load_annotations(name: str) -> tuple[Cameras, list[dict]]:
    """
    """    
    def parse_keyframes(keyframes: dict) -> Cameras:
        """
        """
        H = keyframes['render_height']
        W = keyframes['render_width']
        c2w = []
        fxs = []
        fys = []
        for frame in keyframes['keyframes']:
            camera_to_world = torch.tensor(eval(frame['matrix']))
            camera_to_world = camera_to_world.reshape(4, 4).T[:3, :]
            c2w.append(camera_to_world)
            
            fov = eval(frame['properties'])[0][1]
            focal_length = three_js_perspective_camera_focal_length(fov, H)
            fxs.append(focal_length)
            fys.append(focal_length)
        return Cameras(
            fx=torch.tensor(fxs),
            fy=torch.tensor(fys),
            cx=W / 2,
            cy=H / 2,
            camera_to_worlds=torch.stack(c2w, dim=0),
        )

    data_dir = DATASET_DIR / f'../Localization eval dataset/{name}'
    keyframes = json.load(open(data_dir / 'keyframes.json'))
    cameras = parse_keyframes(keyframes)
    
    annotations = []
    for filename in natsorted(glob(str(data_dir / '*_rgb.json'))):
        annotations.append(json.load(open(filename)))
    return cameras, annotations


def localize(model: DiscreteFeatureField, camera: Cameras, positives: list[str], bbox) -> tuple:
    """
    """
    assert camera.camera_to_worlds.ndim == 2, 'Only one camera is supported'

    camera_original = copy.deepcopy(camera)
    camera = transform_cameras_inverse(camera, *model.renderer.get_camera_transform())
    outputs_ours = model.render(positives, camera, return_relevancy=True)
    outputs_nerf = model.renderer.render_vanilla(camera_original)
    
    #outputs_ours['relevancy'] = \
    #    kornia.filters.median_blur(outputs_ours['relevancy'][None, None], (5, 5))[0][0]
    
    max_value = outputs_ours['relevancy'].max()
    max_index = torch.nonzero(outputs_ours['relevancy'] == max_value)
    #y = max_index[:, 0].float().mean().item()
    #x = max_index[:, 1].float().mean().item()
    y = max_index[:, 0].median().item()
    x = max_index[:, 1].median().item()
    return y, x, {
        'relevancy_ours': outputs_ours['relevancy'  ].cpu().numpy(),
        'relevancy_lerf': outputs_nerf['relevancy_0'].cpu().numpy()[..., 0]
    }


def evaluate_scene(name: str, experiment: str) -> dict:
    """
    """
    def check_position(position: tuple, bbox: list[list]) -> bool:
        x1 = bbox[0][0]
        y1 = bbox[0][1]
        x2 = bbox[1][0]
        y2 = bbox[1][1]
        x, y = position
        return x1 <= x <= x2 and y1 <= y <= y2
        
    print(f'Evaluating localization for scene {name}')

    cameras, annotations = load_annotations(name)
    cameras.rescale_output_resolution(0.25) # annotations uses 270x480 instead of 1080x1920
    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')
    
    model = load_model(name, config)
    stats = {}
    for i, annotation in tqdm(enumerate(annotations)):
        for data in annotation['shapes']:
            positive, bbox = data['label'], data['points'] # [[x1, y1], [x2, y2]]
            # fix bbox annotation issues
            if bbox[0][0] > bbox[1][0]:
                bbox[0][0], bbox[1][0] = bbox[1][0], bbox[0][0]
            if bbox[0][1] > bbox[1][1]:
                bbox[0][1], bbox[1][1] = bbox[1][1], bbox[0][1]
            
            y, x, outputs = localize(model, cameras[i], [positive], bbox)
            position = (x, y)
            key = f'{i}_{positive}'
            stats[key] = check_position(position, bbox)

            print(f'Result: {stats[key]}, Positive: {positive}, Position: {position}, Bbox: {bbox}')
            os.makedirs(f'{experiment}/{name}', exist_ok=True)
            image1 = np.array(visualize_relevancy(outputs['relevancy_ours']))
            image2 = np.array(visualize_relevancy(outputs['relevancy_lerf']))
            visualize_bboxes(image1, [flatten_bbox(bbox)], color=(255, 0, 0)).save(f'{experiment}/{name}/{i}_{positive}_ours.png')
            visualize_bboxes(image2, [flatten_bbox(bbox)], color=(255, 0, 0)).save(f'{experiment}/{name}/{i}_{positive}_lerf.png')
    return stats


if __name__ == '__main__':
    import os
    experiment = 'experiments/localization'
    os.makedirs(experiment, exist_ok=True)
    for scene in ['bouquet', 'figurines', 'ramen', 'teatime', 'waldo_kitchen']:
        path = Path(f'{experiment}/{scene}.json')
        if path.exists():
            continue
        stats = evaluate_scene(scene, experiment)
        with open(path, 'w') as f:
            json.dump(stats, f)