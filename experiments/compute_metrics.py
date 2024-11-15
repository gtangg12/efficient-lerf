import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from sklearn.metrics import precision_recall_curve

from efficient_lerf.data.common import DATASETS, DATASET_DIR
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from experiments.evaluate_existence import load_labels


def compute_norm_sim_score(scores: dict, k: str) -> float:
    """
    """
    return (scores[k] - scores['random']) / (1 - scores['random'])


def precision_recall_curve_from_scores_list(
    scores_ours_list: list[dict], 
    scores_lerf_list: list[dict], 
    labels_list: list[dict], names: list[str], filename: Path | str, fontsize=24
):
    """
    """
    num_plots = len(scores_ours_list)
    aspect_ratio = 1.5  # Stretch x-axis to be 1.5 times the y-axis
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots * aspect_ratio, 6), constrained_layout=True)

    if num_plots == 1:
        axes = [axes] # Ensure axes is always iterable.

    for idx, (scores_ours, scores_lerf, labels, ax) in enumerate(
        zip(scores_ours_list, scores_lerf_list, labels_list, axes)
    ):
        probs_ours = np.array([scores_ours[label] for label in labels])
        probs_lerf = np.array([scores_lerf[label] for label in labels])
        targs = np.array([labels[label] for label in labels])

        precision_ours, recall_ours, _ = precision_recall_curve(targs, probs_ours)
        precision_lerf, recall_lerf, _ = precision_recall_curve(targs, probs_lerf)

        ax.plot(recall_ours, precision_ours, label='Quantized Language Field', color='orange', linewidth=4)
        ax.plot(recall_lerf, precision_lerf, label='LERF', color='purple', linewidth=4)
        ax.fill_between(recall_ours, precision_ours, alpha=0.3, color='orange')
        ax.fill_between(recall_lerf, precision_lerf, alpha=0.3, color='purple')
        ax.set_xlabel('Recall', fontsize=fontsize)
        if idx == 0:
            ax.set_ylabel('Precision', fontsize=fontsize) # Only add y-label to the first plot
            ax.set_yticks([0, 1])
        else:
            ax.set_yticks([]) # Remove y-ticks for subsequent plots
        ax.set_xticks([0, 1])
        ax.tick_params(axis='both', labelsize=20)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_facecolor('aliceblue')

        # Add a title for each subplot
        ax.set_title(f'{names[idx]}', fontsize=fontsize, pad=20)

    # Add a single legend in the first subplot (bottom-left corner)
    axes[0].legend(loc='lower left', fontsize=fontsize, frameon=True)

    # Save the concatenated plots
    fig.savefig(filename, bbox_inches='tight')
    plt.clf()


def compute_accuracy(labels: dict) -> float:
    """
    """
    return sum(labels.values()) / len(labels)


if __name__ == '__main__':
    # # Feature Superpixel Alignment
    # feature_superpixel_alignment = defaultdict(dict)

    # for scene in DATASETS:
    #     with open(f'experiments/feature_superpixel_alignment/{scene}.json', 'r') as f:
    #         scores = json.load(f)
    #     feature_superpixel_alignment['clip'][scene] = compute_norm_sim_score(scores['clip'], k='scale_mean')
    #     feature_superpixel_alignment['dino'][scene] = compute_norm_sim_score(scores['dino'], k='total')
    # feature_superpixel_alignment['clip']['mean'] = np.mean(list(feature_superpixel_alignment['clip'].values()))
    # feature_superpixel_alignment['dino']['mean'] = np.mean(list(feature_superpixel_alignment['dino'].values()))

    # with open('experiments/feature_superpixel_alignment/metrics.json', 'w') as f:
    #     json.dump(feature_superpixel_alignment, f)

    # # Feature Map Alignment
    # feature_map = defaultdict(dict)

    # for scene in DATASETS:
    #     with open(f'experiments/feature_maps/{scene}.json', 'r') as f:
    #         scores = json.load(f)
    #     feature_map['clip'][scene] = compute_norm_sim_score(scores['clip'], k='scale_mean')
    #     feature_map['dino'][scene] = compute_norm_sim_score(scores['dino'], k='dino')
    # feature_map['clip']['mean'] = np.mean(list(feature_map['clip'].values()))
    # feature_map['dino']['mean'] = np.mean(list(feature_map['dino'].values()))

    # with open('experiments/feature_maps/metrics.json', 'w') as f:
    #     json.dump(feature_map, f)
    
    # Existence
    scenes = ['bouquet', 'figurines', 'teatime', 'waldo_kitchen']
    scores_ours_list = []
    scores_lerf_list = []
    labels_list = []
    for scene in scenes:
        with open(f'experiments/existence/{scene}.json', 'r') as f:
            scores = json.load(f)
        reader = LERFFrameSequenceReader(DATASET_DIR, scene)
        labels = load_labels(reader.data_dir)
        scores_ours_list.append(scores['ours'])
        scores_lerf_list.append(scores['lerf'])
        labels_list.append(labels)
    precision_recall_curve_from_scores_list(scores_ours_list, scores_lerf_list, labels_list, scenes,'experiments/existence/metrics.png')
    exit()
    # Localization
    localization = {}
    for scene in ['bouquet', 'figurines', 'ramen', 'teatime', 'waldo_kitchen']:
        with open(f'experiments/localization/{scene}.json', 'r') as f:
            labels = json.load(f)
        localization[scene] = compute_accuracy(labels)
    
    with open('experiments/localization/metrics.json', 'w') as f:
        json.dump(localization, f)