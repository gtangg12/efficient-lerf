import os
import time
from collections import Counter, defaultdict

import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.utils.visualization import *
from efficient_lerf.quantization_methods import *
from efficient_lerf.quantization_interpolation import *


def name_clip(j):
    return f'clip_{j}'

def name_dino():
    return 'dino'


class FeatureMapQuantization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config

    def process_sequence(self, sequence: FrameSequence, renderer: Renderer) -> FrameSequence:
        """
        """
        start_time = time.time()

        sequence_downsampled = downsample(sequence, downsample=self.config.downsample)
        
        pca = defaultdict(dict)
        clip_codebook = []
        dino_codebook = []
        clip_cindices = []
        dino_cindices = []
        clip_count = 0
        dino_count = 0
        for i in range(0, len(sequence), self.config.batch):
    
            print(f'Quantizing feature maps {i} - {i + self.config.batch}')

            batch = sequence_downsampled[i:i + self.config.batch]
            batch = self.quantize(batch, renderer, pca=pca, index=i)
            clip_codebook.append(batch.clip_codebook)
            dino_codebook.append(batch.dino_codebook)
            clip_cindices.append(batch.clip_codebook_indices + clip_count)
            dino_cindices.append(batch.dino_codebook_indices + dino_count)
            clip_count += len(batch.clip_codebook)
            dino_count += len(batch.dino_codebook)

        sequence.clip_codebook = torch.cat(clip_codebook, dim=0)
        sequence.dino_codebook = torch.cat(dino_codebook, dim=0)
        sequence.clip_codebook_indices = torch.cat(clip_cindices, dim=0)
        sequence.dino_codebook_indices = torch.cat(dino_cindices, dim=0)

        # print('Visualizing quantized feature maps')

        # #sequence = upsample(sequence_clone, sequence)

        # for i in tqdm(range(len(sequence))):
        #     if not (self.config.visualize_dir and i % self.config.visualize_stride == 0):
        #         continue
        #     # feature maps upsampled automatically to image size
        #     for j in range(len(renderer.scales)):
        #         quant = sequence.feature_map('clip', i, j)
        #         visualize_features(quant.numpy(), pca[i][name_clip(j)]).save(f'{self.config.visualize_dir}/{name_clip(j)}_{i:003}_quant.png')
        #     quant = sequence.feature_map('dino', i)
        #     visualize_features(quant.numpy(), pca[i][name_dino()]).save(f'{self.config.visualize_dir}/{name_dino()}_{i:003}_quant.png')

        duration = time.time() - start_time
        sequence.metadata['quantization_duration'] = duration
        print(f'Feature map quantization took {duration:.2f} seconds')

        print('Feature map quantization:', len(sequence))
        print('Clip codebook:', sequence.clip_codebook.shape)
        print('Dino codebook:', sequence.dino_codebook.shape)
        print('Clip codebook indices:', sequence.clip_codebook_indices.shape)
        print('Dino codebook indices:', sequence.dino_codebook_indices.shape)

        return sequence

    def quantize(self, sequence: FrameSequence, renderer: Renderer, pca: dict = None, index=None) -> FrameSequence:
        """
        """
        cameras = sequence.transform_cameras(*renderer.get_camera_transform())
        M = len(renderer.scales)
        index = index if index is not None else 0

        accum_embed_means = defaultdict(list)
        accum_assignments = defaultdict(list)
        accum_count = Counter()
        pca = pca if pca is not None else defaultdict(dict)
        if self.config.visualize_dir is not None:
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

            if self.config.visualize_dir and iter % self.config.visualize_stride == 0:
                _pca = compute_pca(embed.numpy(), use_torch=True)
                visualize_features(embed.numpy(), _pca).save(f'{self.config.visualize_dir}/{name}_{iter:003}.png')
                visualize_features(quant.numpy(), _pca).save(f'{self.config.visualize_dir}/{name}_{iter:003}_quant_local.png')
                pca[iter][name] = _pca
            return quant

        def quantize_global(names: list[str], ratio: float) -> tuple:
            """
            Returns codebook: (k, d), codebook_indices: (N, len(names), H, W)
            """
            codebook = []
            cindices = []
            count = 0
            for i, name in tqdm(enumerate(names)):
                _codebook, _cindices = setup_codebook(
                    accum_embed_means[name],
                    accum_assignments[name],
                    k=int(ratio * accum_count[name]) # each scale based on the same superpixels
                )
                codebook.append(_codebook)
                cindices.append(_cindices + count)
                count += len(_codebook)

            # Concat codebooks: M x (k_i, d) -> (k, d)
            codebook = torch.cat(codebook, dim=0)
            # Stack assignments: M x (N, H, W) -> (N, M, H, W)
            cindices = torch.stack(cindices, dim=1)

            for i in range(len(sequence)):
                iter = i + index
                if not (self.config.visualize_dir and iter % self.config.visualize_stride == 0):
                    continue
                for j, name in enumerate(names):
                    quant = codebook[cindices[i, j]]
                    visualize_features(quant.numpy(), pca[iter][name]).save(f'{self.config.visualize_dir}/{name}_{iter:003}_quant_global.png')
            return codebook, cindices
        
        print('Running per frame local quantization')

        for i, camera in tqdm(enumerate(cameras)):
            iter = i + index
            outputs = renderer.render(camera)
            
            image = sequence.images[i]
            if self.config.visualize_dir and iter % self.config.visualize_stride == 0:
                visualize_image(image.numpy()).save(f'{self.config.visualize_dir}/image_{iter:003}.png')

            renderer.enable_model_cache()
            for j, scale in enumerate(renderer.scales):
                embed = renderer.render_scale(camera, scale)
                quantize_local(iter, name_clip(j), image, embed)
            renderer.disable_model_cache()

            quantize_local(iter, name_dino(), image, outputs['dino'])
        
        for k, v in accum_embed_means.items():
            # Concat codebooks: (k_i, d) -> (k, d)
            accum_embed_means[k] = torch.cat(v, dim=0)
            # Stack assignments: (H, W) -> (N, H, W)
            accum_assignments[k] = torch.stack(accum_assignments[k])

        print('Running global quantization')

        clip_codebook, clip_codebook_indices = quantize_global([name_clip(j) for j in range(M)], self.config.k_ratio)
        dino_codebook, dino_codebook_indices = quantize_global([name_dino()                   ], self.config.k_ratio)
        
        sequence.clip_codebook = clip_codebook
        sequence.dino_codebook = dino_codebook
        sequence.clip_codebook_indices = clip_codebook_indices
        sequence.dino_codebook_indices = dino_codebook_indices
        return sequence


if __name__ == '__main__':
    from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
    from efficient_lerf.data.sequence import save_sequence, load_sequence

    reader = LERFFrameSequenceReader('/home/gtangg12/data/lerf/LERF Datasets/', 'bouquet')
    sequence = reader.read(slice=(0, 4, 1))
    renderer = Renderer('/home/gtangg12/efficient-lerf/outputs/bouquet/lerf/2024-11-07_112933/config.yml')

    feature_map_quant = FeatureMapQuantization(OmegaConf.create({
        'batch': 2,
        'downsample': 4,
        'k_ratio': 0.05,
        'superpixels_ncomponents': 2048,
        'superpixels_compactness': 0,
        'visualize_dir': reader.data_dir / 'sequence/visualizations',
        'visualize_stride': 10
    }))
    sequence = feature_map_quant.process_sequence(sequence, renderer)
    print(len(sequence))

    print(sequence.images.shape)
    print(sequence.cameras.shape)
    print(sequence.clip_codebook.shape)
    print(sequence.dino_codebook.shape)
    print(sequence.clip_codebook_indices.shape)
    print(sequence.dino_codebook_indices.shape)
    print(sequence.metadata)

    save_sequence(reader.data_dir / 'sequence', sequence)
    sequence2 = load_sequence(reader.data_dir / 'sequence')

    assert torch.allclose(sequence.images, sequence2.images)
    assert torch.allclose(sequence.cameras.camera_to_worlds, sequence2.cameras.camera_to_worlds)
    assert torch.allclose(sequence.clip_codebook, sequence2.clip_codebook)
    assert torch.allclose(sequence.dino_codebook, sequence2.dino_codebook)
    assert torch.allclose(sequence.clip_codebook_indices, sequence2.clip_codebook_indices)
    assert torch.allclose(sequence.dino_codebook_indices, sequence2.dino_codebook_indices)
    print(sequence2.metadata)