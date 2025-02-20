import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.models.model_netvlad import ModelNetVLAD
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.visualization import *
from efficient_lerf.quantization_methods import *


class FeatureMapQuantization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        if self.config.k_adaptive:
            self.model_netvlad = ModelNetVLAD()

    def process_sequence(self, sequence: FrameSequence, renderer: Renderer) -> FrameSequence:
        """
        """
        self.config.sequence_path = sequence.metadata['data_dir'] / 'sequence' / 'sequence.pt'
        self.config.visualize_dir = sequence.metadata['data_dir'] / 'sequence' / 'visualizations'

        start_time = time.time()

        sequence_downsampled = sequence.clone()
        sequence_downsampled.rescale_camera_resolution(scale=1 / self.config.downsample)
        
        pca = defaultdict(dict)
        codebook_vectors = defaultdict(list)
        codebook_indices = defaultdict(list)
        counts = Counter()
        for i in range(0, len(sequence), self.config.batch):
            print(f'Quantizing feature maps {i} - {i + self.config.batch}')
            
            batch = sequence_downsampled[i:i + self.config.batch]
            batch = self.quantize(batch, renderer, pca=pca, index=i)
            for name in renderer.feature_names():
                codebook_vectors[name].append(batch.codebook_vectors[name])
                codebook_indices[name].append(batch.codebook_indices[name] + counts[name])
                counts[name] += len(batch.codebook_vectors[name])

        for name in renderer.feature_names():
            sequence.codebook_vectors[name] = torch.cat(codebook_vectors[name], dim=0)
            sequence.codebook_indices[name] = torch.cat(codebook_indices[name], dim=0)

        duration = time.time() - start_time
        sequence.metadata['quantization_duration'] = duration
        print(f'Feature map quantization took {duration:.2f} seconds')

        print('Feature map quantization:', len(sequence))
        for name in renderer.feature_names():
            print(f'{name} codebook vectors:', sequence.codebook_vectors[name].shape)
            print(f'{name} codebook indices:', sequence.codebook_indices[name].shape)

        return sequence

    def quantize(self, sequence: FrameSequence, renderer: Renderer, pca: dict = None, index=None) -> FrameSequence:
        """
        """
        sequence = sequence.clone()
        sequence.transform_cameras(*renderer.get_camera_transform())
        index = index if index is not None else 0
        
        if self.config.k_adaptive:
            visual_embeds = self.model_netvlad(sequence)
            visual_compactness = torch.mean(visual_embeds @ visual_embeds.T)

        accum_embed_means = defaultdict(list)
        accum_assignments = defaultdict(list)
        accum_count = Counter()
        pca = pca if pca is not None else defaultdict(dict)
        if self.config.visualize is not None:
            os.makedirs(self.config.visualize_dir, exist_ok=True)

        def quantize_local(
            iter: int,
            name: str,
            image: TorchTensor['H', 'W'],
            embed: TorchTensor['H', 'W', 'd'], 
        ) -> TorchTensor['H', 'W', 'd']:
            """
            Returns quant: (H, W, d)
            """
            embed = norm(embed.detach().cpu(), -1)
            embed_mean, assignment = quantize_image_superpixel(
                image, 
                embed, 
                self.config.superpixels_ncomponents, 
                self.config.superpixels_compactness,
            )
            accum_embed_means[name].append(embed_mean)
            accum_assignments[name].append(assignment + accum_count[name])
            accum_count[name] += len(torch.unique(assignment))
            quant = embed_mean[assignment]

            if self.config.visualize and iter % self.config.visualize_stride == 0:
                _pca = compute_pca(embed.numpy(), use_torch=True)
                visualize_features(embed.numpy(), _pca).save(f'{self.config.visualize_dir}/{name}_{iter:003}.png')
                visualize_features(quant.numpy(), _pca).save(f'{self.config.visualize_dir}/{name}_{iter:003}_quant_local.png')
                pca[iter][name] = _pca
            return quant

        def quantize_global(names: list[str]) -> tuple:
            """
            Returns codebook: (k, d), codebook_indices: (N, len(names), H, W)
            """            
            codebook_vectors = []
            codebook_indices = []
            count = 0
            for i, name in tqdm(enumerate(names)):
                k = int(self.config.k_ratio * accum_count[name])
                if self.config.k_adaptive:
                    r = 1 / len(sequence)
                    k = r * k + (1 - r) * (1 - visual_compactness) * k # linear interpolation between min clusters and fixed ratio
                _codebook_vectors, _codebook_indices = setup_codebook(
                    accum_embed_means[name],
                    accum_assignments[name],
                    k=k # each scale based on the same superpixels
                )
                codebook_vectors.append(_codebook_vectors)
                codebook_indices.append(_codebook_indices + count)
                count += len(_codebook_vectors)

            # Concat codebooks: M x (k_i, d) -> (k, d)
            codebook_vectors = torch.cat(codebook_vectors, dim=0)
            # Stack assignments: M x (N, H, W) -> (N, M, H, W)
            codebook_indices = torch.stack(codebook_indices, dim=1)

            for i in range(len(sequence)):
                iter = i + index
                if not (self.config.visualize and iter % self.config.visualize_stride == 0):
                    continue
                for j, name in enumerate(names):
                    quant = codebook_vectors[codebook_indices[i, j]]
                    visualize_features(quant.numpy(), pca[iter][name]).save(f'{self.config.visualize_dir}/{name}_{iter:003}_quant_global.png')
            return codebook_vectors, codebook_indices
        
        print('Running per frame local quantization')

        local_quantization_path = self.config.sequence_path.parent / f'local_quantization_{index}.pt'
        if local_quantization_path.exists():
            accum_embed_means, accum_assignments, accum_count, pca = torch.load(local_quantization_path)
        else:
            for i, camera in tqdm(enumerate(sequence.cameras)):
                iter = i + index

                image = sequence.images[i]
                if self.config.visualize and iter % self.config.visualize_stride == 0:
                    visualize_image(image.numpy()).save(f'{self.config.visualize_dir}/image_{iter:003}.png')

                for name in renderer.feature_names():
                    for j, embed in enumerate(renderer.render(name, camera)):
                        quantize_local(iter, name + f'_{j}', image, embed)
            
            for k, v in accum_embed_means.items():
                # Concat codebooks: (k_i, d) -> (k, d)
                accum_embed_means[k] = torch.cat(v, dim=0)
                # Stack assignments: (H, W) -> (N, H, W)
                accum_assignments[k] = torch.stack(accum_assignments[k])

            if self.config.cache_local_quantization_outputs:
                torch.save((accum_embed_means, accum_assignments, accum_count, pca), local_quantization_path)

        print('Running global quantization')

        for name, nscales in renderer.feature_names().items():
            codebook_vectors, codebook_indices = quantize_global([name + f'_{j}' for j in range(nscales)])
            sequence.codebook_vectors[f'{name}'] = codebook_vectors
            sequence.codebook_indices[f'{name}'] = codebook_indices
        
        return sequence


if __name__ == '__main__':
    from efficient_lerf.data.common import DATASET_DIR
    from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
    from efficient_lerf.data.sequence import save_sequence, load_sequence
    from efficient_lerf.renderer.renderer_lerf import LERFRenderer

    tests = Path('tests') / 'sequence'
    os.makedirs(tests, exist_ok=True)

    reader = LERFFrameSequenceReader('bouquet')
    sequence = reader.read(slice=(0, 4, 1))
    sequence.metadata['data_dir'] = tests.parent
    renderer = LERFRenderer('bouquet')

    feature_map_quant = FeatureMapQuantization(OmegaConf.create({
        'batch': 2,
        'downsample': 4,
        'k_ratio': 0.05,
        'k_adaptive': False,
        'superpixels_ncomponents': 2048,
        'superpixels_compactness': 5,
        'visualize': True,
        'visualize_stride': 10,
        'cache_local_quantization_outputs': True
    }))
    sequence = feature_map_quant.process_sequence(sequence, renderer)
    print(len(sequence))

    print(sequence.images.shape)
    print(sequence.cameras.shape)
    for name in renderer.feature_names():
        print(name, 'codebook vectors', sequence.codebook_vectors[name].shape)
        print(name, 'codebook indices', sequence.codebook_indices[name].shape)
    print(sequence.metadata)

    save_sequence(tests / 'sequence.pt', sequence)
    sequence2 = load_sequence(tests / 'sequence.pt')

    assert torch.allclose(sequence.images, sequence2.images)
    assert torch.allclose(sequence.cameras.camera_to_worlds, sequence2.cameras.camera_to_worlds)
    for name in renderer.feature_names():
        assert torch.allclose(sequence.codebook_vectors[name], sequence2.codebook_vectors[name])
        assert torch.allclose(sequence.codebook_indices[name], sequence2.codebook_indices[name])
    print(sequence2.metadata)