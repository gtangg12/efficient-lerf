import yaml
import json
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from efficient_lerf.data.common import DATASET_DIR, CONFIGS_DIR
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.visualization import *
from efficient_lerf.feature_field import DiscreteFeatureField, load_model


def load_labels(data_dir: Path | str) -> dict:
    """
    """
    labels = {}
    try:
        with open(data_dir / 'labels.yaml', 'r') as f:
            labels_tail = yaml.safe_load(f)
            labels.update(labels_tail)
    except FileNotFoundError: # LERF Dataset inconsistent
        pass
    try:
        with open(data_dir / 'coco_labels.yaml', 'r') as f:
            labels_coco = yaml.safe_load(f)
            labels.update(labels_coco)
    except FileNotFoundError:
        pass
    return labels


def exist_sequence(model: DiscreteFeatureField, positives: list[str]) -> dict:
    """
    """
    return model.find_clip(positives)


def exist_renderer(model: DiscreteFeatureField, positives: list[str]) -> dict:
    """
    """
    sequence = model.sequence
    renderer = model.renderer

    cameras = sequence.transform_cameras(*renderer.get_camera_transform())

    renderer.pipeline.image_encoder.set_positives(positives)

    scores = defaultdict(lambda: 0)
    relevancy_maps = defaultdict(list)
    for camera in tqdm(cameras):
        outputs = renderer.render_vanilla(camera)
        for i in range(len(positives)):
            positive = positives[i]
            relevancy_map = outputs[f'relevancy_{i}'].squeeze(-1) # (H, W)
            scores[positive] = max(scores[positive], relevancy_map.max().item())
            relevancy_maps[positive].append(relevancy_map.cpu().numpy())
    for k, v in relevancy_maps.items():
        relevancy_maps[k] = np.stack(v)
    return scores, relevancy_maps


def evaluate_scene(name: str, experiment: Path | str) -> dict:
    """
    """
    thresholds = np.linspace(0.5, 1, 11)
    min_union = 128

    print(f'Evaluating existence for scene {name}')

    reader = LERFFrameSequenceReader(DATASET_DIR, name)
    labels = load_labels(reader.data_dir)
    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

    model = load_model(name, config)
    positives = list(labels.keys())
    scores_ours, relevancy_maps_ours = exist_sequence(model, positives)
    scores_lerf, relevancy_maps_lerf = exist_renderer(model, positives)
    
    iou_t = Counter()
    iou_t_counts = Counter()
    iou_t_max = 0
    iou_t_counts_max = 0
    for positive, exist in labels.items():
        if not exist:
            continue
        print(f'{positive}: {scores_ours[positive]:.2f} (ours) vs {scores_lerf[positive]:.2f} (LERF)')
        os.makedirs(f'{experiment}/visualizations/{name}/{positive}', exist_ok=True)
        
        for i in range(len(model.sequence)):
            map_ours = relevancy_maps_ours[positive][i]
            map_lerf = relevancy_maps_lerf[positive][i]
            iou_max = None
            for threshold in thresholds:
                map_ours_t = np.array(map_ours)
                map_lerf_t = np.array(map_lerf)
                map_ours_t[map_ours < threshold] = 0
                map_lerf_t[map_lerf < threshold] = 0
                cap = ((map_ours_t > 0) & (map_lerf_t > 0)).sum()
                cup = ((map_ours_t > 0) | (map_lerf_t > 0)).sum()
                if cup > min_union:
                    iou = cap / cup
                    iou_t[threshold] += iou
                    iou_t_counts[threshold] += 1
                    if iou_max is None or iou > iou_max:
                        iou_max = iou
                visualize_relevancy(map_ours_t).save(f'{experiment}/visualizations/{name}/{positive}/{i}@{threshold:.2f}_ours.png')
                visualize_relevancy(map_lerf_t).save(f'{experiment}/visualizations/{name}/{positive}/{i}@{threshold:.2f}_lerf.png')
            if iou_max is not None:
                iou_t_max += iou_max
                iou_t_counts_max += 1
    
    for t in thresholds:
        iou_t[t] = iou_t[t] / iou_t_counts[t] if iou_t_counts[t] > 0 else 0
        print(f'IoU@{t}: {iou_t[t]}')
    iou_t_max = iou_t_max / iou_t_counts_max if iou_t_counts_max > 0 else 0
    print(f'IoU@max: {iou_t_max}')
    iou_t['max'] = iou_t_max

    return scores_ours, relevancy_maps_ours, \
           scores_lerf, relevancy_maps_lerf, iou_t


if __name__ == '__main__':
    import os
    experiment = 'experiments/existence'
    os.makedirs(experiment, exist_ok=True)
    for scene in ['bouquet', 'figurines', 'teatime', 'waldo_kitchen']: # LERF Dataset does not release table scene
        path = Path(f'{experiment}/{scene}.json')
        if path.exists():
           continue
        scores_ours, relevancy_maps_ours,\
        scores_lerf, relevancy_maps_lerf, iou_t = evaluate_scene(scene, experiment)
        with open(path, 'w') as f:
            json.dump({
                'ours': scores_ours, 
                'lerf': scores_lerf}, f, indent=4
            )
        with open(f'{experiment}/{scene}_iou.json', 'w') as f:
            json.dump(iou_t, f, indent=4)