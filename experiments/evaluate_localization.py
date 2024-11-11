from nerfstudio.cameras.cameras import Cameras
from lerf.encoders.clip_encoder import CLIPNetwork

from efficient_lerf.data.sequence import FrameSequencePointCloud


def localize(pc: FrameSequencePointCloud, camera: Cameras, positive: str, model: CLIPNetwork) -> tuple:
    """
    """
    model.set_positives([positive])
    
    outputs = pc.render(camera)
    coords = outputs['coords'].flatten(0, -2)
    scores = model.get_relevancy(outputs['clip'].flatten(0, -2), positive_id=0)
    return coords[scores.argmax()]