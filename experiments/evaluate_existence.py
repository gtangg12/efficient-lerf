import yaml
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from efficient_lerf.data.common import DATASET_DIR, CONFIGS_DIR
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.visualization import *
from efficient_lerf.quantization_model import DiscreteFeatureField, load_model


def exist_sequence(model: DiscreteFeatureField, positives: list[str], device='cuda') -> dict:
    """
    """
    return model.exist(positives)


def exist_renderer(model: DiscreteFeatureField, positives: list[str], device='cuda') -> dict:
    """
    """
    sequence = model.sequence
    renderer = model.renderer

    cameras = sequence.transform_cameras(*renderer.get_camera_transform())

    renderer.pipeline.image_encoder.set_positives(positives)

    scores = defaultdict(lambda: 0)
    for camera in tqdm(cameras):
        outputs = renderer.render_vanilla(camera)
        for i in range(len(positives)):
            positive = positives[i]
            scores[positive] = max(scores[positive], outputs[f'relevancy_{i}'].max().item())
    return scores


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


def evaluate_scene(name: str) -> dict:
    """
    """
    print(f'Evaluating existence for scene {name}')

    reader = LERFFrameSequenceReader(DATASET_DIR, name)
    labels = load_labels(reader.data_dir)
    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

    model = load_model(name, config)
    scores_ours = exist_sequence(model, list(labels.keys()))
    scores_lerf = exist_renderer(model, list(labels.keys()))

    #precision_recall_curve_from_scores(scores_ours, scores_lerf, labels, f'precision_recall_curve_{name}.png')
    return scores_ours, scores_lerf


if __name__ == '__main__':
    import os
    experiment = 'experiments/existence'
    os.makedirs(experiment, exist_ok=True)
    for scene in ['bouquet', 'figurines', 'teatime', 'waldo_kitchen']: # LERF Dataset does not release table scene
        path = Path(f'{experiment}/{scene}.json')
        if path.exists():
            continue
        scores_ours, scores_lerf = evaluate_scene(scene)
        with open(path, 'w') as f:
            json.dump({
                'ours': scores_ours, 
                'lerf': scores_lerf}, f, indent=4
            )