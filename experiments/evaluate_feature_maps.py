import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.math import norm, mean

from experiments.common import DATASETS, RENDERERS, summarize, setup


SAVE_DIR = Path('experiments/feature_maps')


def distance(embed_targ: TorchTensor['H', 'W', 'dim'], embed_pred: TorchTensor['H', 'W', 'dim']) -> float:
    """
    """
    distances = torch.sum(embed_targ * embed_pred, dim=-1)
    return torch.mean(distances).item()


def distance_baseline(embed_targ: TorchTensor['H', 'W', 'dim']) -> float:
    """
    """
    embed_pred = embed_targ.mean(dim=(0, 1))
    return distance(embed_targ, embed_pred)


from efficient_lerf.utils.visualization import *
from efficient_lerf.utils.math import *

def evaluate_feature(name: str, sequence: FrameSequence, renderer: Renderer) -> dict:
    """
    """
    sequence = sequence.clone()
    sequence.transform_cameras(*renderer.get_camera_transform())
    
    stats = defaultdict(list)

    for i, camera in tqdm(enumerate(sequence.cameras)):
        for j, embed in enumerate(renderer.render(name, camera)):
            embed_pred = sequence.feature_map(name, i, j, upsample=True) # match original resolution
            embed_pred = norm(embed_pred , dim=-1)
            embed_targ = norm(embed.cpu(), dim=-1)
            stats[j].append(distance(embed_targ, embed_pred))
            stats['baseline'].append(distance_baseline(embed_targ))
    
    stats = {k: mean(v) for k, v in stats.items()}
    stats['scale_mean'] = mean([v for k, v in stats.items() if k != 'baseline'])
    return stats


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type, stride=20) -> dict:
    """
    """
    stats, path, renderer_name = setup(SAVE_DIR, scene, RendererT)
    if stats is not None:
        return stats
    print(f'Evaluating feature maps for renderer {renderer_name} for scene {scene}')
    
    reader, renderer = FrameSequenceReaderT(scene), RendererT(scene)
    sequence = load_sequence(reader.data_dir / 'sequence/sequence.pt')[::stride]

    stats = {}
    for feature_name in renderer.feature_names():
        stats[feature_name] = evaluate_feature(feature_name, sequence, renderer)
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


if __name__ == '__main__':
    accum = {}
    for RendererT, FrameSequenceReaderT in RENDERERS:
        for scene in DATASETS:
            accum[(scene, RendererT)] = evaluate(scene, RendererT, FrameSequenceReaderT)
    summarize(SAVE_DIR, accum)