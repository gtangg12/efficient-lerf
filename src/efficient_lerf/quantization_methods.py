import multiprocessing as mp

import cv2
import numpy as np
import torch
import faiss
import fast_slic

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.utils.math import norm


def quantize_embed_kmeans(embeds: TorchTensor['N', 'dim'], k: int) -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (n)
    """
    embeds = norm(embeds, dim=-1)
    embeds = embeds.numpy()
    kmeans = faiss.Kmeans(d=embeds.shape[-1], k=k, spherical=True, niter=5, verbose=False, min_points_per_centroid=1) # min points to suppress warning
    kmeans.train(embeds)
    # kmeans = KMeans(n_clusters=k, mode='cosine', verbose=1)
    # codebook_indices = kmeans.fit_predict(embeds)
    
    codebook = kmeans.centroids
    codebook = torch.from_numpy(codebook)
    _, codebook_indices = kmeans.index.search(embeds, 1)
    codebook_indices = torch.from_numpy(codebook_indices).squeeze(1) # Remove extra dimension
    return codebook.cpu(), codebook_indices.cpu().int()


def quantize_embed_kmeans_heirarchical(embeds: TorchTensor['N', 'dim'], k: int, levels: int) -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (n)
    """
    embeds = norm(embeds, dim=-1)
    embeds = embeds.numpy()

    n, d = embeds.shape
    b = int(np.floor(k ** (1 / levels)))

    print(f'Heirarchical kmeans with branching factor {b}, levels {levels}')

    # Initialize cluster assignments
    codebook = []
    codebook_indices = np.zeros(n, dtype=int)
    clusters = 0

    def hierarchical_kmeans(indices, level):
        # Assign unique cluster id to leaf clusters
        print(f'Level {level}, {len(indices)} samples')
        nonlocal clusters
        if level == levels:
            codebook_indices[indices] = clusters
            clusters += 1
            return

        # Perform k-means on current cluster
        embeds_current = embeds[indices]
        kmeans = faiss.Kmeans(d=d, k=b, spherical=True, niter=5, verbose=False, min_points_per_centroid=1) # min points to suppress warning
        kmeans.train(embeds_current)
        _, codebook_indices_current = kmeans.index.search(embeds_current, 1)
        codebook_indices_current = codebook_indices_current[:, 0, ...]

        # Recurse on child clusters
        for i in range(b):
            if level == levels - 1:
                codebook.append(kmeans.centroids[i])
            child_indices = indices[codebook_indices_current == i]
            hierarchical_kmeans(child_indices, level + 1)

    indices = np.arange(n)
    hierarchical_kmeans(indices, level=0)
    codebook = torch.from_numpy(np.array(codebook))
    codebook_indices = torch.from_numpy(codebook_indices)
    return codebook.cpu(), codebook_indices.cpu().int()


def setup_codebook(embeds: TorchTensor['N', 'dim'], assignments: TorchTensor['H', 'W'], k: int, method='kmeans') -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (N)
    """
    assert method in ['kmeans', 'kmeans_heirarchical']
    func = quantize_embed_kmeans if method == 'kmeans' else \
           quantize_embed_kmeans_heirarchical
    if len(embeds) == k:
        return embeds, torch.arange(k)[assignments]
    codebook, indices = func(embeds.flatten(0, -2), k)
    return codebook, indices[assignments]


def search_codebook(embeds: TorchTensor['N', 'dim'], codebook: TorchTensor['k', 'dim']) -> TorchTensor['N']:
    """
    Returns: indices: (N)
    """
    embeds = norm(embeds, dim=-1).numpy()
    index = faiss.IndexFlatIP(codebook.shape[-1])
    index.add(codebook.numpy())
    _, indices = index.search(embeds, 1)
    return torch.from_numpy(indices).squeeze(1)


def compute_superpixels(image: TorchTensor['H', 'W', 3], ncomponents=1024, compactness=10) -> TorchTensor['H', 'W']:
    """
    Returns: superpixel assignments: (H, W)
    """
    image = image.numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    slic = fast_slic.Slic(num_components=ncomponents, compactness=compactness)
    return torch.from_numpy(slic.iterate(image)).int()


def quantize_image_superpixel(
    image: TorchTensor['H', 'W', 3], 
    embed: TorchTensor['H', 'W', 'dim'], ncomponents=1024, compactness=10
) -> tuple:
    """
    Returns: embed_mean: (k, d), assignemnts: (h, w)
    """
    assignment = compute_superpixels(image, ncomponents, compactness)

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


def quantize_image_superpixel_codebook(
    image: TorchTensor['H', 'W'],
    embed: TorchTensor['H', 'W', 'd'], 
    cbook: TorchTensor['k', 'd'], 
    ncomponents=1024, compactness=10
):
    """
    Returns: assignment_cindices: (h, w)
    """
    embed_mean, assignment = quantize_image_superpixel(image, embed, ncomponents, compactness)
    embed_mean_cindices = search_codebook(embed_mean, cbook)
    assignment_cindices = embed_mean_cindices[assignment] # Remap superpixel indices to codebook indices
    return assignment_cindices


if __name__ == '__main__':
    image = torch.load('/home/gtangg12/efficient-lerf/tests/lerf/tensors/rgb.pt')
    embed = torch.load('/home/gtangg12/efficient-lerf/tests/lerf/tensors/clip.pt')
    image = (image * 255).int()

    from efficient_lerf.utils.visualization import *

    embed_mean, assignment = quantize_image_superpixel(image, embed, ncomponents=2048, compactness=10)
    print(embed_mean.shape, assignment.shape)

    codebook, codebook_indices = quantize_embed_kmeans(embed_mean, k=512)
    print(codebook.shape, codebook_indices.shape)
    assert len(codebook_indices.unique()) == len(codebook)

    codebook, codebook_indices = quantize_embed_kmeans_heirarchical(embed_mean, k=512, levels=2)
    print(codebook.shape, codebook_indices.shape)
    assert len(codebook_indices.unique()) == len(codebook)