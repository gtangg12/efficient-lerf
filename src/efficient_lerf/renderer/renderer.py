from abc import ABC, abstractmethod
from pathlib import Path

from efficient_lerf.data.common import TorchTensor
from nerfstudio.cameras.cameras import Cameras


class Renderer(ABC):
    """
    """
    def __init__(self, checkpoint: Path | str, device='cuda'):
        """
        """
        self.checkpoint = Path(checkpoint)
        self.device = device

    @abstractmethod
    def feature_names(self) -> dict:
        """
        Returns names of features and the respective number of scales.
        """

    @abstractmethod
    def get_camera_transform(self) -> tuple[TorchTensor[4, 4], float]:
        """
        Returns camera transform and scale from input to renderer space.
        """

    def render(self, name: str, camera: Cameras) -> iter:
        """
        Returns an iterator over the rendered features `name` for the given `camera`.
        """
        return eval(f'self.render_{name}')(camera)

    def find(self, name: str, positives: any, features: TorchTensor[..., 'dim']) -> TorchTensor['N', '...']:
        """
        Returns dict of scores for each positive with relevancy maps.
        """
        return eval(f'self.find_{name}')(positives, features)