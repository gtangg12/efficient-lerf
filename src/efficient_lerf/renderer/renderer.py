from abc import ABC, abstractmethod

from efficient_lerf.data.common import TorchTensor
from nerfstudio.cameras.cameras import Cameras


class Renderer(ABC):
    """
    """
    @abstractmethod
    def feature_names(self) -> dict:
        """
        Returns names of features and the respective number of scales.
        """

    @abstractmethod
    def get_train_cameras(self) -> Cameras:
        """
        Returns cameras used during renderer training.
        """

    @abstractmethod
    def get_camera_transform(self) -> tuple[TorchTensor[4, 4], float]:
        """
        Returns camera transform and scale from input to renderer space.
        """

    def render_features(self, name: str, camera: Cameras) -> iter:
        """
        Returns an iterator over the rendered features `name` for the given `camera`.
        """
        return eval(f'self.render_{name}')(camera)