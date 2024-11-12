import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.visualization import *


def exist_sequence(sequence: FrameSequence, renderer: Renderer, positives: list[str], device='cuda') -> dict:
    """
    """
    renderer.pipeline.image_encoder.set_positives(positives)

    scores = defaultdict(lambda: 0)
    for i in range(len(positives)):
        positive = positives[i]
        probs = renderer.pipeline.image_encoder.get_relevancy(sequence.clip_codebook.to(device), positive_id=i)
        scores[positive] = probs[:, 0].max().item() # positive prob
    return scores


def exist_renderer(renderer: Renderer, positives: list[str], device='cuda') -> dict:
    """
    """
    cameras = sequence.transform_cameras(*renderer.get_camera_transform())

    renderer.pipeline.image_encoder.set_positives(positives)

    scores = defaultdict(lambda: 0)
    for camera in tqdm(cameras):
        outputs = renderer.render_vanilla(camera)
        for i in range(len(positives)):
            positive = positives[i]
            scores[positive] = max(scores[positive], outputs[f'relevancy_{i}'].max().item())
    print(scores)
    exit()
    return scores


def load_labels(data_dir: Path | str) -> dict:
    """
    """
    labels = {}
    with open(data_dir / 'labels.yaml', 'r') as f:
        labels_tail = yaml.safe_load(f)
    with open(data_dir / 'coco_labels.yaml', 'r') as f:
        labels_coco = yaml.safe_load(f)
    labels.update(labels_tail)
    labels.update(labels_coco)
    return labels


def precision_recall_curve_from_scores(scores: dict, target: dict, filename: Path | str):
    """
    """
    scores = np.array([scores[label] for label in scores])
    target = np.array([target[label] for label in scores])

    precision, recall, _ = precision_recall_curve(target, scores)
    
    plt.plot(recall, precision, marker='.', label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig(filename)


if __name__ == '__main__':
    from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader

    reader = LERFFrameSequenceReader('/home/gtangg12/data/lerf/LERF Datasets/', 'bouquet')
    sequence = load_sequence(reader.data_dir / 'sequence')
    print(len(sequence))

    renderer = Renderer('/home/gtangg12/efficient-lerf/outputs/bouquet/lerf/2024-11-07_112933/config.yml')

    labels = load_labels(reader.data_dir)
    
    exist_sequence(sequence, renderer, list(labels.keys()))
    #exist_renderer(renderer, list(labels.keys()))