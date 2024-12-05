import cv2
import numpy as np
import torch
import plotly.graph_objects as go
from PIL import Image
from nerfstudio.utils.colormaps import ColormapOptions, apply_colormap, apply_pca_colormap

from efficient_lerf.data.common import NumpyTensor
from efficient_lerf.utils.math import compute_pca


BBox = tuple[int, int, int, int] # TLBR format


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
GREEN = (  0, 255,   0)
BLUE  = (  0,   0, 255)


def visualize_tiles(tiles: list[NumpyTensor['h', 'w', 3]], r: int, c: int) -> Image.Image:
    """
    """
    h, w = tiles[0].shape[:2]
    assert all(tile.shape[:2] == (h, w) for tile in tiles), 'tiles must have the same shape'
    image = np.concatenate([
        np.concatenate(tiles[cindex * c:(cindex + 1) * c], axis=1)
        for cindex in range(r)
    ], axis=0)
    return Image.fromarray(image.astype(np.uint8))


def visualize_image(image: NumpyTensor['h', 'w', 3]) -> Image.Image:
    """
    """
    return Image.fromarray(image.astype(np.uint8))


def visualize_depth(depth: NumpyTensor['h', 'w'], percentile=99) -> Image.Image:
    """
    """
    depth = np.clip(depth, 0, np.percentile(depth, percentile))
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = np.clip(depth, 0, 1)
    depth = np.stack([depth] * 3, axis=-1)
    return Image.fromarray((depth * 255).astype(np.uint8))


def visualize_features(features: NumpyTensor['h', 'w', 'dim'], pca=None, valid=None, background=BLACK) -> Image.Image:
    """
    Given features, visualizes features' first 3 (RGB) principal components.
    """
    H, W, _ = features.shape
    if pca is None:
        pca = compute_pca(features, n=3, use_torch=True)
    features = torch.from_numpy(features)
    image = apply_pca_colormap(features, pca, ignore_zeros=False).numpy()
    image.reshape(H, W, 3)
    return Image.fromarray((image * 255).astype('uint8'))


def visualize_relevancy(score: NumpyTensor['h', 'w']) -> Image.Image:
    """
    """
    H, W = score.shape
    probs = torch.from_numpy(np.clip(score - 0.5, 0, 1))
    probs = probs.flatten().unsqueeze(1)
    image = apply_colormap(probs / (probs.max() + 1e-6), ColormapOptions('turbo')).numpy()
    image = image.reshape(H, W, 3)
    return Image.fromarray((image * 255).astype('uint8'))


def visualize_bboxes(image: NumpyTensor['h', 'w', 3], bboxes: list[BBox], color=GREEN) -> Image.Image:
    """
    """
    image_bboxes = image.copy()
    for bbox in bboxes:
        cv2.rectangle(image_bboxes, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bboxes)


def visualize_outline(cmask: NumpyTensor['h', 'w'], image: NumpyTensor['h', 'w', 3], color=RED) -> Image.Image:
    """
    """
    image = image.copy()

    kernel = np.ones((3, 3), dtype=np.uint8)

    outline = np.zeros_like(cmask, dtype=np.uint8)
    for label in np.unique(cmask):
        gradient = cv2.morphologyEx((cmask == label).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
        outline = np.maximum(outline, gradient)
    image[outline == 1] = color
    return image


def visualize_point_cloud(points, depths=None, colors=None, size=1, depth_percentile=0.8, nsamples=None) -> go.Figure:
    """
    """
    colors = np.full(points.shape, 0.5) if colors is None else colors / 255.0
    
    if depths is not None:
        valid = (depths != 0) & (depths < np.percentile(depths, depth_percentile * 100))
        points = points[valid]
        depths = depths[valid]
        colors = colors[valid]
    
    if nsamples is not None and points.shape[0] > nsamples:
        indices = np.random.choice(points.shape[0], nsamples, replace=False)
        points = points[indices]
        depths = depths[indices] if depths is not None else None
        colors = colors[indices]

    trace = go.Scatter3d(
        x=points[:, 0], 
        y=points[:, 1], 
        z=points[:, 2],
        mode='markers',
        marker=dict(size=size, color=colors, opacity=0.8)
    )
    layout = go.Layout(
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            zaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
        )
    )
    fig = go.Figure(data=[trace], layout=layout)
    return fig