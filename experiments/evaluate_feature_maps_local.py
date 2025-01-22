import json
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, NullLocator, FormatStrFormatter
from torchvision.transforms import PILToTensor, Compose
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.quantization_methods import quantize_image_patch, quantize_image_superpixel
from efficient_lerf.utils.math import compute_pca, mean, norm
from efficient_lerf.utils.visualization import *

from experiments.common import DATASETS, DATASETS_SUBSET, RENDERERS, setup


SAVE_DIR = Path('experiments/outputs/feature_maps_local')


lpips_evaluator = LearnedPerceptualImagePatchSimilarity()
lpips_transform = Compose([
    PILToTensor(),
    lambda x: x.unsqueeze(0) / 255 * 2 - 1
])

def lpips(sample: Image.Image, target: Image.Image):
    return lpips_evaluator(
        lpips_transform(sample),
        lpips_transform(target),
    )


def key(method: str, scale: int, params: dict) -> str:
    return f'{method}@{scale}@{str(params)}'


def distance(
    image: TorchTensor['H', 'W', 3], 
    embed: TorchTensor['H', 'W', 'dim'],
    embed_visual: Image.Image, 
    pca: TorchTensor['K', 'dim'],
    method: callable
) -> tuple:
    """
    """
    embed_mean, assignment = method(image, embed)
    quant = embed_mean[assignment]
    quant_visual = visualize_features(quant.cpu().numpy(), pca=pca)
    loss_cosine = torch.mean(torch.sum(embed * quant, dim=-1)).item()
    loss_lpips = lpips(quant_visual, embed_visual).item()
    return loss_cosine, loss_lpips, len(embed_mean), quant_visual


def evaluate_feature(name: str, sequence: FrameSequence, renderer: Renderer, methods: dict, path=None, rescale=0.25) -> dict:
    """
    """
    sequence = sequence.clone()
    sequence.transform_cameras(*renderer.get_camera_transform())
    sequence.rescale_camera_resolution(rescale) # match renderer resolution

    stats = defaultdict(list)

    for i, (camera, image) in tqdm(enumerate(zip(sequence.cameras, sequence.images))):
        for j, embed in enumerate(renderer.render(name, camera)):
            embed_np = embed.cpu().numpy()
            pca = compute_pca(embed_np, use_torch=True)
            embed_visual = visualize_features(embed_np, pca=pca)
            if i == 0 and j == 0 and path is not None:
                embed_visual.save(path / f'embed_{name}_{i}_{j}.png')

            for method in methods.keys():
                method_fn = globals()[f'quantize_image_{method}']
                for params in methods[method]['params']:
                    losses_cosine, losses_lpips, n, quant_visual = \
                        distance(image, embed, embed_visual, pca, lambda x, y: method_fn(x, y, **params))
                    stats[key(method, j, params)].append(
                        (losses_cosine, losses_lpips, n)
                    )
                    if i == 0 and j == 0 and path is not None:
                        quant_visual.save(path / f'embed_{name}_{i}_{j}_{method}_{str(params)}.png')
    
    scale_mean = defaultdict(list)
    for k, v in stats.items():
        method, _, params = tuple(k.split('@'))
        scale_mean[key('scale_mean', method, params)].extend(v)

    stats = {}
    for k, v in scale_mean.items():
        losses_cosine, losses_lpips, length = zip(*v)
        stats[k] = (mean(losses_cosine), mean(losses_lpips), mean(length))
        #print(f'{k}: {stats}')
    return stats


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type, num_samples=20) -> dict:
    """
    """
    stats, path, renderer_name = setup(SAVE_DIR, scene, RendererT)
    if stats is not None:
       return stats

    reader, renderer = FrameSequenceReaderT(scene), RendererT(scene)
    sequence = reader.read()
    sequence = sequence[::len(sequence) // num_samples]
    
    print(f'Evaluating feature maps for renderer {renderer_name} for scene {scene} on {len(sequence)} frames')

    stats = {}
    for feature_name in renderer.feature_names():
        stats[feature_name] = evaluate_feature(feature_name, sequence, renderer, methods={
            'patch': {
                'params': [
                    {'patch_size':  2},
                    {'patch_size':  4},
                    {'patch_size':  8},
                    {'patch_size': 12},
                    {'patch_size': 16},
                    {'patch_size': 20},
                    {'patch_size': 24},
                    {'patch_size': 32},
                ]
            },
            'superpixel': {
                'params': [
                    {'ncomponents': 8192, 'compactness': 0},
                    {'ncomponents': 4096, 'compactness': 0},
                    {'ncomponents': 2048, 'compactness': 0},
                    {'ncomponents': 1024, 'compactness': 0},
                    {'ncomponents':  512, 'compactness': 0},
                    {'ncomponents':  256, 'compactness': 0},
                    {'ncomponents':  128, 'compactness': 0},
                    {'ncomponents':   64, 'compactness': 0},
                ]
            }
        }, path=path)
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


def plot_comparison_curve(accum: dict, filename: Path | str, group: int) -> None:
    """
    """
    nplot = len(accum)
    nrows = group
    ncols = nplot // group
    feature_names = set(accum[list(accum.keys())[0]].keys())

    method2color = {'patch': 'purple', 'superpixel': 'orange'}
    metric2plots = {'cosine': None, 'lpips': None}

    for feature_name in feature_names:
        for metric in metric2plots.keys():
            fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 5 * nrows), constrained_layout=True) # 1.5 aspect ratio
            if nrows == 1:
                axes = np.array([axes]) # Ensure axes iterable
            metric2plots[metric] = (fig, axes)

        for i, ((scene, RendererT), stats) in enumerate(accum.items()):
            x = defaultdict(lambda: defaultdict(list))
            y = defaultdict(lambda: defaultdict(list))
            for k, v in stats[feature_name].items():
                _, method, params = tuple(k.split('@'))
                x['cosine'][method].append(v[2])
                y['cosine'][method].append(v[0])
                x['lpips' ][method].append(v[2])
                y['lpips' ][method].append(v[1])

            r = i // ncols
            c = i  % ncols
            for metric, (fig, axes) in metric2plots.items():
                for method in x[metric].keys():
                    axes[r, c].plot(
                        x[metric][method], 
                        y[metric][method], 
                        label=method, color=method2color[method], linewidth=4
                    )
                axes[r, c].set_title(scene)
                axes[r, c].set_facecolor('aliceblue')
                axes[r, c].set_xscale('log')
                axes[r, c].set_xlabel('Codebook Size')
                if i == 0:
                    axes[r, c].set_ylabel('Quantization Similarity')
                axes[r, c].set_ylim(None if metric == 'cosine' else 0, 1)
                axes[r, c].xaxis.set_minor_locator(NullLocator())
                axes[r, c].yaxis.set_major_locator(LinearLocator(5))
                axes[r, c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

        for metric, (fig, axes) in metric2plots.items():
            axes[-1, -1].legend(loc='lower right', framealpha=1)

            fig.savefig(f'{filename}/{feature_name}{group}_{metric}.png')


if __name__ == '__main__':
    for RendererT, FrameSequenceReaderT in RENDERERS:
        accum = {}
        for scene in DATASETS:
            stats = evaluate(scene, RendererT, FrameSequenceReaderT)
            accum[(scene, RendererT)] = stats
        accum1 = {}
        accum2 = {}
        for k, v in accum.items():
            if k[0] in DATASETS_SUBSET:
                accum1[k] = v
            else:
                accum2[k] = v
        plot_comparison_curve(accum1, SAVE_DIR / f'{RendererT.__name__}', group=1)
        plot_comparison_curve(accum2, SAVE_DIR / f'{RendererT.__name__}', group=2)