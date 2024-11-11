import torch
from lerf.encoders.clip_encoder import CLIPNetwork

from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.renderer.renderer import Renderer


def exist(sequence: FrameSequence, positive: str, threshold: float, model: CLIPNetwork) -> bool:
    """
    """
    model.set_positives([positive])
    scores = model.get_relevancy(sequence.clip_codebook, positive_id=0) # index of positive in positives
    return torch.max(scores) > threshold


def exist(renderer: Renderer, positive: str, threshold: float, model: CLIPNetwork) -> bool:
    """
    """
    model.set_positives([positive])

    best_score = 0
    for camera in renderer.cameras:
        score = 0
        for scale in renderer.scales:
            embed = renderer.renderer_scale(camera, scale)
            score = max(score, model.get_relevancy(embed, positive_id=0).max())
        best_score = max(best_score, score)
    return best_score > threshold