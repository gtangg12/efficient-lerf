from collections import defaultdict

import torch
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.renderer.renderer import Renderer


def mean(x):
    return torch.mean(torch.tensor(x))


def evaluate_clip(sequence: FrameSequence, renderer: Renderer) -> dict:
    """
    """
    stats = defaultdict(list)
    
    for i, camera in enumerate(sequence.cameras):
        for j, scale in enumerate(renderer.scales):
            embed_targ = renderer.renderer_scale(camera, scale)
            embed_pred = sequence.clip_codebook[sequence.clip_codebook_idx[i, j]]
            stats[scale].append(
                torch.sum(embed_targ * embed_pred) / (camera.height * camera.width)
            )
    stats = {k: mean(v) for k, v in stats.items()}
    stats['total'] = mean(list(stats.values()))
    return stats


def evaluate_dino(sequence: FrameSequence, renderer: Renderer) -> dict:
    """
    """
    stats = defaultdict(list)
    
    for i, camera in enumerate(sequence.cameras):
        embed_targ = renderer.render(camera)['dino']
        embed_pred = sequence.dino_codebook[sequence.dino_codebook_idx[i]]
        stats['dino'].append(
            torch.sum(embed_targ * embed_pred) / (camera.height * camera.width)
        )
    stats = {k: mean(v) for k, v in stats.items()}
    return stats