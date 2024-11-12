import cv2
import numpy as np
import plotly.graph_objects as go
from PIL import Image

from efficient_lerf.data.common import NumpyTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.utils.math import compute_pca, min_max_norm


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


def visualize_features(features: NumpyTensor['h', 'w', 'dim'], pca=None) -> Image.Image:
    """
    Given features, visualizes features' first 3 (RGB) principal components.
    """
    H, W, dim = features.shape
    features = features.reshape(-1, dim)
    if pca is None:
        pca = compute_pca(features, n=3)
    pca_features = pca.transform(features)
    pca_features = min_max_norm(pca_features, dim=-1)
    pca_features = pca_features.reshape(H, W, 3)
    return Image.fromarray((pca_features * 255).astype('uint8'))


def visualize_bbox(bbox: BBox, image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
    """
    """
    image_bbox = image.copy()
    cv2.rectangle(image_bbox, (bbox[1], bbox[0]), (bbox[3], bbox[2]), color, 2)
    return Image.fromarray(image_bbox)


def visualize_bboxes(bboxes: list[BBox], image: NumpyTensor['h', 'w', 3], color=GREEN) -> Image.Image:
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


def visualize_sequence(sequence: FrameSequence) -> Image.Image:
    """
    """
    def render(visualize_func: callable, data: NumpyTensor) -> NumpyTensor:
        return [np.array(visualize_func(x)) for x in data]
    
    images = render(visualize_image, sequence.images)
    num_cols = len(images)
    num_rows = 1
    tiles = images
    if sequence.depths is not None:
        depths = render(visualize_depth, sequence.depths)
        num_rows += 1
        tiles.extend(depths)
    if sequence.embeds_clip is not None:
        features = render(visualize_features, sequence.embeds_clip)
        num_rows += 1
        tiles.extend(features)
    if sequence.embeds_dino is not None:
        features = render(visualize_features, sequence.embeds_dino)
        num_rows += 1
        tiles.extend(features)
    return visualize_tiles(tiles, num_rows, num_cols)