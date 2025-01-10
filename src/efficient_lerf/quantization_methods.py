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


def setup_codebook(embeds: TorchTensor['N', 'dim'], assignments: TorchTensor['H', 'W'], k: int, method='kmeans') -> tuple:
    """
    Returns: codebook: (k, d), codebook_indices: (N)
    """
    assert method in ['kmeans', 'kmeans_heirarchical']
    # func = quantize_embed_kmeans if method == 'kmeans' else \
    #        quantize_embed_kmeans_heirarchical
    func = quantize_embed_kmeans
    if len(embeds) == k:
        return embeds, torch.arange(k)[assignments]
    codebook, indices = func(embeds.flatten(0, -2), k)
    return codebook, indices[assignments]


def compute_superpixels(image: TorchTensor['H', 'W', 3], ncomponents=1024, compactness=10) -> TorchTensor['H', 'W']:
    """
    Returns: superpixel assignments: (H, W)
    """
    image = image.numpy().astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    slic = fast_slic.Slic(num_components=ncomponents, compactness=compactness)
    return torch.from_numpy(slic.iterate(image)).int()


def quantize_image_superpixel(image: TorchTensor['H', 'W', 3], embed: TorchTensor['H', 'W', 'dim'], ncomponents=1024, compactness=10) -> tuple:
    """
    Returns: embed_mean: (k, d), assignments: (h, w)
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
    embed_mean = norm(embed_mean, dim=-1)
    return embed_mean, assignment


def quantize_image_patch(image: TorchTensor['H', 'W', 3], embed: TorchTensor['H', 'W', 'dim'], patch_size=4) -> tuple:
    """
    Returns: embed_mean: (k, d), assignemnts: (h, w). Requires image for consistent interface.
    """
    H, W, D = embed.shape
    
    embed = torch.nn.functional.pad(
        embed, (
            0, 0,
            0, patch_size - embed.shape[1] % patch_size, 
            0, patch_size - embed.shape[0] % patch_size,
        )
    )
    patches_h = embed.shape[0] // patch_size
    patches_w = embed.shape[1] // patch_size
    embed = embed.view(patches_h, patch_size, patches_w, patch_size, D)
    
    embed_mean = embed.mean(dim=(1, 3)).view(-1, D)
    embed_mean = norm(embed_mean, dim=-1)

    rindices = torch.arange(H, device=embed.device).unsqueeze(1).expand(-1, W)
    cindices = torch.arange(W, device=embed.device).unsqueeze(0).expand(H, -1)

    assignment = (rindices // patch_size) * patches_w + (cindices // patch_size) # (padH, padW)
    assignment = assignment[:H, :W] # (H, W)

    return embed_mean, assignment


if __name__ == '__main__':
    import os
    from pathlib import Path
    from torchvision.transforms import PILToTensor
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from efficient_lerf.utils.math import *
    from efficient_lerf.utils.visualization import *

    path = Path('/home/gtangg12/efficient-lerf/tests/quant_methods')
    os.makedirs(path, exist_ok=True)

    lpips = LearnedPerceptualImagePatchSimilarity()

    def to_torch(image: Image.Image):
        return PILToTensor()(image).unsqueeze(0) / 255 * 2 - 1

    image = torch.load(path.parent / 'lerf/tensors/rgb.pt')
    embed = torch.load(path.parent / 'lerf/tensors/clip.pt')
    embed = norm(embed, dim=-1)
    image = (image * 255).int()
    pca = compute_pca(embed, use_torch=True)
    embed_image = visualize_features(embed.numpy(), pca=pca)
    embed_image.save(path / 'embed.png')
    print(lpips(to_torch(embed_image), to_torch(embed_image))) # lower lpips better

    embed_mean, assignment = quantize_image_superpixel(image, embed, ncomponents=2048, compactness=5)
    quant = embed_mean[assignment]
    quant_image = visualize_features(quant.numpy(), pca=pca)
    quant_image.save(path / 'quant_superpixel.png')
    print(lpips(to_torch(embed_image), to_torch(quant_image)))
    print(embed_mean.shape, assignment.shape)
    print('Reconstruction error:', torch.mean(torch.sum(embed * quant, dim=-1)).item())

    codebook, codebook_indices = quantize_embed_kmeans(embed_mean, k=64)
    print(codebook.shape, codebook_indices.shape)
    assert len(codebook_indices.unique()) == len(codebook)

    # codebook, codebook_indices = quantize_embed_kmeans_heirarchical(embed_mean, k=512, levels=2)
    # print(codebook.shape, codebook_indices.shape)
    # assert len(codebook_indices.unique()) == len(codebook)

    embed_mean, assignment = quantize_image_patch(image, embed, patch_size=4)
    quant = embed_mean[assignment]
    quant_image = visualize_features(quant.numpy(), pca=pca)
    quant_image.save(path / 'quant_patch.png')
    print(lpips(to_torch(embed_image), to_torch(quant_image)))
    print(embed_mean.shape, assignment.shape)
    print('Reconstruction error:', torch.mean(torch.sum(embed * quant, dim=-1)).item())