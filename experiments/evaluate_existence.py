import json
import os
import yaml
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve

from efficient_lerf.data.common import DATASET_DIR, CONFIGS_DIR
from efficient_lerf.utils.math import compute_relevancy
from efficient_lerf.utils.visualization import *
from efficient_lerf.feature_field import VQFeatureField, load_model

from experiments.common import DATASETS_SUBSET, RENDERERS, setup


SAVE_DIR = Path('experiments/outputs/existence')


def load_labels(data_dir: Path | str) -> dict:
    """
    Load LERF dataset annotations.
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


def exist_sequence(model: VQFeatureField, labels: dict, feature_name='clip', path: Path | str = None) -> dict:
    """
    """
    scores_tensor, relevancy_maps_tensor = model.find(feature_name, labels.keys())
    scores = {}
    for j, (positive, exist) in enumerate(labels.items()):
        scores[positive] = scores_tensor[j].item()
        if path is not None and exist:
            for i, relevancy_map in enumerate(relevancy_maps_tensor[j]):
                visualize_relevancy(relevancy_map.cpu().numpy()).save(f'{path}/visualizations/{positive}_{i}_ours.png')
    return scores


def exist_renderer(model: VQFeatureField, labels: dict, feature_name='clip', rescale=0.25, path: Path | str = None) -> dict:
    """
    """
    sequence = model.sequence
    renderer = model.renderer

    sequence = sequence.clone()
    sequence.transform_cameras(*renderer.get_camera_transform())
    sequence.rescale_camera_resolution(rescale) # match renderer resolution

    scores = defaultdict(lambda: 0)

    for i, camera in tqdm(enumerate(sequence.cameras)):
        probs = []
        for embed in renderer.render(feature_name, camera):
            probs.append(renderer.find(feature_name, labels.keys(), embed))
        probs = torch.stack(probs, dim=1)
        embed_scores = probs.amax(dim=[1, 2, 3])
        embed_relevancy_maps = compute_relevancy(probs, threshold=0.5)
        for j, (positive, exist) in enumerate(labels.items()):
            scores[positive] = max(scores[positive], embed_scores[j].item())
            if path is not None and exist:
                visualize_relevancy(embed_relevancy_maps[j].cpu().numpy()).save(f'{path}/visualizations/{positive}_{i}_test.png')
    return scores


def evaluate_scene(scene: str, RendererT: type, FrameSequenceReaderT: type) -> dict:
    """
    """
    labels = load_labels(DATASET_DIR / f'lerf/LERF Datasets/{scene}')

    scores, path, renderer_name = setup(SAVE_DIR, scene, RendererT, name='scores.json')
    if scores is not None:
        return scores, labels
    os.makedirs(f'{path}/visualizations', exist_ok=True)
    print(f'Evaluating feature maps for renderer {renderer_name} for scene {scene}')

    config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

    model = load_model(scene, config, RendererT, FrameSequenceReaderT)
    #model.sequence = model.sequence[::200]
    scores_ours = exist_sequence(model, labels, path=path)
    scores_test = exist_renderer(model, labels, path=path)
    
    for positive, exist in labels.items():
        if exist: 
            print(f'{positive}: {scores_ours[positive]:.2f} (ours) vs {scores_test[positive]:.2f} ({renderer_name})')
    scores = {
        'ours': scores_ours,
        'test': scores_test,
    }
    with open(f'{path}/scores.json', 'w') as f:
        json.dump(scores, f)
    return scores, labels


def plot_precision_recall_curve(accum_scores: dict, accum_labels: dict, name: str, filename: Path | str) -> None:
    """
    """
    nplots = len(accum_scores)
    fig, axes = plt.subplots(1, nplots, figsize=(7.5 * nplots, 5), constrained_layout=True) # 1.5 aspect ratio
    if nplots == 1:
        axes = [axes] # Ensure axes iterable

    for i, (scene, outputs) in enumerate(accum_scores.items()):
        labels = accum_labels[scene]
        probs_ours = np.array([outputs['ours'][label] for label in labels])
        probs_test = np.array([outputs['test'][label] for label in labels])
        targs = np.array([labels[label] for label in labels])

        precision_ours, recall_ours, _ = precision_recall_curve(targs, probs_ours)
        precision_test, recall_test, _ = precision_recall_curve(targs, probs_test)

        axes[i].plot(recall_ours, precision_ours, label='Ours', color='orange', linewidth=4)
        axes[i].plot(recall_test, precision_test, label=name  , color='purple', linewidth=4)
        axes[i].fill_between(recall_ours, precision_ours, alpha=0.3, color='orange')
        axes[i].fill_between(recall_test, precision_test, alpha=0.3, color='purple')

        if i == 0:
            axes[i].set_ylabel('Precision')
            axes[i].set_yticks([0, 1])
        else:
            axes[i].set_yticks([])
        axes[i].set_xlabel('Recall')
        axes[i].set_xticks([0, 1])

        axes[i].tick_params(axis='both')
        axes[i].set_xlim(0, 1)
        axes[i].set_ylim(0, 1)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top']  .set_visible(False)
        axes[i].set_facecolor('aliceblue')
        axes[i].set_title(f'{scene}')

    axes[0].legend(loc='lower left', framealpha=1)

    if filename is None:
        plt.show()
    fig.savefig(filename, bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for RendererT, FrameSequenceReaderT in RENDERERS:
        accum_scores = {}
        accum_labels = {}
        for scene in DATASETS_SUBSET:
            scores, labels = evaluate_scene(scene, RendererT, FrameSequenceReaderT)
            accum_scores[scene] = scores
            accum_labels[scene] = labels
        renderer_name = RendererT.__name__
        plot_precision_recall_curve(
            accum_scores, 
            accum_labels, 
            renderer_name.strip('Renderer'), SAVE_DIR / f'{renderer_name}.png'
        )