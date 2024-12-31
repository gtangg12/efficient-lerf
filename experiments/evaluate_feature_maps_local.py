import json
from collections import defaultdict
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tqdm import tqdm

from efficient_lerf.data.common import DATASET_DIR
from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.quantization_methods import quantize_image_patch, quantize_image_superpixel
from efficient_lerf.utils.math import mean, norm

from experiments.common import DATASETS, RENDERERS, setup


SAVE_DIR = Path('experiments/feature_maps_local')


def key(scale: int, method: str, params: dict) -> str:
    return f'{scale}@{method}@{str(params)}'


def evaluate_feature(name: str, sequence: FrameSequence, renderer: Renderer, methods: dict, rescale=0.25) -> dict:
    """
    """
    sequence = sequence.clone()
    sequence.transform_cameras(*renderer.get_camera_transform())
    sequence.rescale_camera_resolution(rescale) # match renderer resolution
    
    stats = defaultdict(list)
    stats_baseline = []

    for i, (camera, image) in tqdm(enumerate(zip(sequence.cameras, sequence.images))):
        for j, embed in enumerate(renderer.render(name, camera)):
            for method in methods.keys():
                method_fn = globals()[f'quantize_image_{method}']
                for params in methods[method]['params']:
                    embed = norm(embed.detach().cpu(), -1)
                    embed_mean, assignment = method_fn(image, embed, **params)
                    quant_loss = torch.mean(torch.sum(embed * embed_mean[assignment], dim=-1)).item()
                    stats[key(j, method, params)].append(
                        (quant_loss, len(embed_mean))
                    )

                    # Compute baseline
                    embed_mean_baseline = embed.reshape(-1, embed.shape[-1]).mean(0)
                    quant_loss_baseline = torch.mean(torch.sum(embed * embed_mean_baseline, dim=-1)).item()
                    stats_baseline.append(quant_loss_baseline)
    
    scale_mean = defaultdict(list)
    for k, v in stats.items():
        _, method, params = tuple(k.split('@'))
        scale_mean[key('scale_mean', method, params)].extend(v)

    stats_combined = {}
    for k, v in scale_mean.items():
        losses, length = zip(*v)
        stats_combined[k] = (mean(losses), mean(length))
        #print(f'{k}: {stats_combined[k]}')
    stats_combined['baseline'] = mean(stats_baseline)
    return stats_combined


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type, stride=20) -> dict:
    """
    """
    stats, path, renderer_name = setup(SAVE_DIR, scene, RendererT)
    if stats is not None:
        return stats
    print(f'Evaluating feature maps for renderer {renderer_name} for scene {scene}')

    reader, renderer = FrameSequenceReaderT(scene), RendererT(scene)
    sequence = load_sequence(reader.data_dir / 'sequence/sequence.pt')[::stride]
    
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
        })
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


def plot_comparison_curve(accum: dict, path: Path | str) -> None:
    """
    """
    nplots = len(accum)
    feature_names = set(accum[list(accum.keys())[0]].keys())

    baselines = {}
    for feature_name in feature_names:
        baselines[feature_name] = []
        for (scene, RendererT), stats in accum.items():
            baselines[feature_name].append(stats[feature_name].pop('baseline'))
    
    colors = {'patch': 'purple', 'superpixel': 'orange'}

    for feature_name in feature_names:
        fig, axes = plt.subplots(1, nplots, figsize=(7.5 * nplots, 5), constrained_layout=True) # 1.5 aspect ratio
        if nplots == 1:
            axes = [axes] # Ensure axes iterable

        for i, ((scene, RendererT), stats) in enumerate(accum.items()):
            x = defaultdict(list)
            y = defaultdict(list)
            baseline = baselines[feature_name][i]
            for k, v in stats[feature_name].items():
                _, method, _ = tuple(k.split('@'))
                x[method].append(v[1])
                y[method].append(v[0])
            for method in x.keys():
                axes[i].plot(x[method], y[method], label=method, color=colors[method], linewidth=4)
            axes[i].plot([0, max(x['patch'])], [baseline, baseline], label='feature map mean', color='green', linewidth=4, linestyle='--')

            axes[i].set_title(scene)
            axes[i].set_facecolor('aliceblue')
            axes[i].set_xscale('log')
            axes[i].set_xlabel('Codebook Size')
            if i == 0:
                axes[i].set_ylabel('Quantization Similarity')
            axes[i].set_ylim(None, 1)
            axes[i].yaxis.set_major_locator(LinearLocator(5))
            axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

        axes[-1].legend(loc='lower right', framealpha=1)
        
        plt.savefig(f'{path}/{RendererT.__name__}/{feature_name}.png')
        plt.clf()


if __name__ == '__main__':
    for RendererT, FrameSequenceReaderT in RENDERERS:
        accum = {}
        for scene in DATASETS:
            stats = evaluate(scene, RendererT, FrameSequenceReaderT)
            accum[(scene, RendererT)] = stats
        plot_comparison_curve(accum, SAVE_DIR)