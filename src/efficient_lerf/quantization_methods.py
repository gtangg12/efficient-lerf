import cv2
import numpy as np
import torch
import faiss
import fast_slic
from fast_pytorch_kmeans import KMeans

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.math import norm


def compute_superpixels(image: TorchTensor['H', 'W', 3], num_components=1024, compactness=10) -> TorchTensor['H', 'W']:
    """
    """
    image = image.numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    slic = fast_slic.Slic(num_components=num_components, compactness=compactness)
    return torch.from_numpy(slic.iterate(image)).long()


def quantize_image_superpixel(image: TorchTensor['H', 'W', 3], embed: TorchTensor['H', 'W', 'dim'], **kwargs) -> tuple:
    """
    Returns: embed_mean: (k, d), assignemnts: (h, w)
    """
    assignment = compute_superpixels(image, **kwargs)

    # Compute mean of embeddings for each superpixel
    dim = embed.shape[-1]
    flat_embeddings = embed.flatten(0, -2)
    flat_assignment = assignment.flatten()
    
    num_labels = flat_assignment.max().item() + 1
    label_sums = torch.zeros(num_labels, dim, device=embed.device)
    label_sums = label_sums.index_add(0, flat_assignment, flat_embeddings)
    label_cnts = torch.bincount(flat_assignment, minlength=num_labels).unsqueeze(1)
    embed_mean = label_sums / label_cnts.clamp_min(1)
    '''
    # check equal compared to naive implementation
    embed_mean_naive = torch.zeros_like(embed_mean)
    for i in range(num_labels):
        embed_mean_naive[i] = flat_embeddings[flat_assignment == i].mean(0)
    assert torch.allclose(embed_mean, embed_mean_naive)
    '''
    return embed_mean, assignment


def quantize_embed_kmeans(embeds: TorchTensor['N', 'dim'], k: int) -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (n)
    """
    embeds = norm(embeds, dim=-1)
    embeds = embeds.numpy()
    kmeans = faiss.Kmeans(d=embeds.shape[-1], k=k, spherical=True, niter=5, verbose=True)
    kmeans.train(embeds)
    # kmeans = KMeans(n_clusters=k, mode='cosine', verbose=1)
    # codebook_indices = kmeans.fit_predict(embeds)
    codebook = kmeans.centroids

    codebook = torch.from_numpy(codebook)
    _, codebook_indices = kmeans.index.search(embeds, 1)
    codebook_indices = torch.from_numpy(codebook_indices).squeeze(1) # Remove extra dimension
    return codebook.cpu(), codebook_indices.cpu()


def quantize_embed_LBG(embeds: TorchTensor['N', 'dim']) -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (n)
    """
    pass


def setup_codebook(embeds: TorchTensor, assignments: TorchTensor, k: int, method='kmeans') -> tuple:
    """
    """
    assert method in ['kmeans', 'LBG']
    func = quantize_embed_kmeans if method == 'kmeans' else \
           quantize_embed_LBG
    if len(embeds) == k:
        return embeds, torch.arange(k)[assignments]
    codebook, indices = func(embeds.flatten(0, -2), k)
    return codebook, indices[assignments]


if __name__ == '__main__':
    image = torch.load('/home/gtangg12/efficient-lerf/tests/lerf/tensors/rgb.pt')
    embed = torch.load('/home/gtangg12/efficient-lerf/tests/lerf/tensors/clip.pt')
    image = (image * 255).long()

    from efficient_lerf.utils.visualization import *

    embed_mean, assignment = quantize_image_superpixel(image, embed, num_components=2048, compactness=10)
    print(embed_mean.shape, assignment.shape)

    codebook, codebook_indices = quantize_embed_kmeans(embed_mean, k=512)
    print(codebook.shape, codebook_indices.shape)
