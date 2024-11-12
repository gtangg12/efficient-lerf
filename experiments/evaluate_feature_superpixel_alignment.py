from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from nerfstudio.cameras.cameras import Cameras

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.math import mean, norm
from efficient_lerf.quantization_methods import compute_superpixels
from efficient_lerf.utils.visualization import *


def patch_sim(embeds: TorchTensor['N', 'dim']) -> float:
    """
    """
    N = embeds.shape[0]
    embeds = norm(embeds.flatten(0, -2), dim=-1)
    sim = torch.matmul(embeds, embeds.T) # N x N
    return 1 / (N * (N - 1)) * torch.sum(sim - torch.eye(N, device=embeds.device)).item()


def patch_sim_random(embeds: TorchTensor['N', 'dim'], nrandom=1024) -> float:
    """
    """
    N = embeds.shape[0]
    embeds = norm(embeds.flatten(0, -2), dim=-1)
    embeds = embeds[torch.randperm(N)[:nrandom]]
    return patch_sim(embeds)


def evaluate_clip_alignment(sequence: FrameSequence, renderer: Renderer, **kwargs) -> dict:
    """
    """
    stats = defaultdict(list)

    cameras = sequence.transform_cameras(*renderer.get_camera_transform())

    for image, camera in tqdm(zip(sequence.images, cameras), 'Evaluating LERF alignment'):
        assignment = compute_superpixels(image, **kwargs).flatten()
        labels = torch.unique(assignment)

        for scale in renderer.scales:
            embed = renderer.render_scale(camera, scale)
            for label in labels:
                stats[scale].append(patch_sim(embed[assignment == label]))
            stats['random'].append(patch_sim_random(embed)) # sample random pairs for baseline

    stats = {k: mean(v) for k, v in stats.items()}
    stats['scale_mean'] = mean([v for k, v in stats.items() if k != 'random'])
    return stats


def evaluate_dino_alignment(sequence: FrameSequence, renderer: Renderer, **kwargs) -> dict:
    """
    """
    stats = defaultdict(list)

    cameras = sequence.transform_cameras(*renderer.get_camera_transform())

    for image, camera in tqdm(zip(sequence.images, cameras), 'Evaluating DINO alignment'):
        assignment = compute_superpixels(image, **kwargs).flatten()
        labels = torch.unique(assignment)

        embed = renderer.render(camera)['dino']
        for label in labels:
            stats['total'].append(patch_sim(embed[assignment == label]))
            stats['random'].append(patch_sim_random(embed))

    stats = {k: mean(v) for k, v in stats.items()}
    return stats


def evaluate_scene(name: str) -> dict:
    """
    """
    pass


if __name__ == '__main__':
    import json
    from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader

    reader = LERFFrameSequenceReader('/home/gtangg12/data/lerf/LERF Datasets/', 'book_store')
    sequence = reader.read(slice=(0, -1, 5))
    renderer = Renderer('/home/gtangg12/efficient-lerf/outputs/book_store/lerf/2024-11-07_102606/config.yml')

    stats = evaluate_clip_alignment(sequence, renderer)
    print(json.dumps(stats, indent=4))

    stats = evaluate_dino_alignment(sequence, renderer)
    print(json.dumps(stats, indent=4))