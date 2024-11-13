import json
from collections import defaultdict

import torch
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor, DATASET_DIR, DATASETS, load_checkpoint
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.math import norm, mean


def distance(embed_targ: TorchTensor['N', 'dim'], embed_pred: TorchTensor['N', 'dim']) -> float:
    """
    """
    embed_targ = norm(embed_targ.flatten(0, -2), dim=-1)
    embed_pred = norm(embed_pred.flatten(0, -2), dim=-1)
    return torch.mean(torch.sum(embed_targ * embed_pred, dim=-1)).item()


def distance_random(embed_targ: TorchTensor['N', 'dim'], embed_pred: TorchTensor['N', 'dim'], nrandom=1024) -> float:
    """
    """
    embed_targ = norm(embed_targ.flatten(0, -2), dim=-1)
    embed_pred = norm(embed_pred.flatten(0, -2), dim=-1)
    embed_targ = embed_targ[torch.randperm(embed_targ.shape[0])[:nrandom]]
    embed_pred = embed_pred[torch.randperm(embed_pred.shape[0])[:nrandom]]
    return distance(embed_targ, embed_pred)


def evaluate_clip(sequence: FrameSequence, renderer: Renderer) -> dict:
    """
    """
    cameras = sequence.transform_cameras(*renderer.get_camera_transform())
    
    stats = defaultdict(list)

    for i, camera in tqdm(enumerate(cameras)):

        renderer.enable_model_cache()
        
        for j, scale in enumerate(renderer.scales):
            embed_targ = renderer.render_scale(camera, scale).cpu()
            embed_pred = sequence.clip_codebook[sequence.clip_codebook_indices[i, j]]
            stats[scale].append(distance(embed_targ, embed_pred))
            stats['random'].append(distance_random(embed_targ, embed_pred))
        
        renderer.disable_model_cache()
    
    stats = {k: mean(v) for k, v in stats.items()}
    stats['scale_mean'] = mean(list(stats.values()))
    return stats


def evaluate_dino(sequence: FrameSequence, renderer: Renderer) -> dict:
    """
    """
    cameras = sequence.transform_cameras(*renderer.get_camera_transform())
    
    stats = defaultdict(list)

    for i, camera in tqdm(enumerate(cameras)):
        embed_targ = renderer.render(camera)['dino'].cpu()
        embed_pred = sequence.dino_codebook[sequence.dino_codebook_indices[i]]
        stats['dino'].append(distance(embed_targ, embed_pred))
        stats['random'].append(distance_random(embed_targ, embed_pred))

    stats = {k: mean(v) for k, v in stats.items()}
    return stats


def evaluate_scene(name: str) -> dict:
    """
    """
    print(f'Evaluating feature maps for scene {name}')
    
    reader = LERFFrameSequenceReader(DATASET_DIR, name)
    sequence = load_sequence(reader.data_dir / 'sequence')
    renderer = Renderer(load_checkpoint(name))
    stats_clip = evaluate_clip(sequence, renderer)
    stats_dino = evaluate_dino(sequence, renderer)
    return stats_clip, stats_dino


if __name__ == '__main__':
    import os
    experiment = 'experiments/feature_maps'
    os.makedirs(experiment, exist_ok=True)
    for scene in DATASETS:
        stats_clip, stats_dino = evaluate_scene(scene)
        with open(f'{experiment}/{scene}.json', 'w') as f:
            json.dump({
                'clip': stats_clip, 
                'dino': stats_dino}, f, indent=4
            )