import torch
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


def compute_pca(features: NumpyTensor, n=3) -> PCA:
    """
    """
    features = features.reshape(-1, features.shape[-1])
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