import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from efficient_lerf.data.common import NumpyTensor, TorchTensor


def mean(x: list):
    return sum(x) / len(x)


def norm(tensor: TorchTensor, dim=None):
    """
    """
    return tensor / tensor.norm(dim=dim, keepdim=True)


def min_max_norm(tensor: NumpyTensor, dim=None):
    """
    """
    tmin = tensor.min(axis=dim, keepdims=True)
    tmax = tensor.max(axis=dim, keepdims=True)
    return (tensor - tmin) / (tmax - tmin)


def compute_pca(features: NumpyTensor, n=3, use_torch=False) -> PCA | TorchTensor:
    """
    """
    features = features.reshape(-1, features.shape[-1])
    if use_torch:
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        _, _, pca = torch.pca_lowrank(features, q=n)
    else:
        pca = PCA(n_components=n)
        pca.fit(features)
    return pca


def pad_poses(poses: TorchTensor['...', 3, 4]) -> TorchTensor['...', 4, 4]:
    """
    """
    output = torch.eye(4).repeat(*poses.shape[:-2], 1, 1)
    output[..., :3, :] = poses[..., :3, :]
    return output


def transpose_list(x: list[list]):
    """
    """
    return list(map(list, zip(*x)))


def upsample_feature_map(x: TorchTensor['H', 'W', 'dim'], upH, upW):
    """
    """
    x = x.permute(2, 0, 1)
    x = F.interpolate(x[None].float(), (upH, upW), mode='nearest')[0].to(x.dtype)
    x = x.permute(1, 2, 0)
    return x


def compute_relevancy(probs: TorchTensor['N', 'M', 'H', 'W'], threshold: float) -> TorchTensor['N', 'H', 'W']:
    """
    """
    return probs.max(1)[0] > threshold