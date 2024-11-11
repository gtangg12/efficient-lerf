import itertools
import os
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from efficient_lerf.data.common import TorchTensor
from efficient_lerf.data.sequence import FrameSequence, save_sequence, load_sequence
from efficient_lerf.models.model_netvlad import ModelNetVLAD
from efficient_lerf.renderer.renderer import Renderer
from efficient_lerf.quantization_methods import *
from efficient_lerf.utils.math import compute_pca
from efficient_lerf.utils.visualization import *


class CameraTrajQuantization:
    """
    """
    def __init__(self, config: OmegaConf):
        """
        """
        self.config = config
        self.module = ModelNetVLAD()

    def process_sequence(self, sequence: FrameSequence) -> tuple[FrameSequence, TorchTensor]:
        """
        """
        embeds = self.module(sequence) # NetVLAD image embeddings

        indices = [0]
        current_embed = embeds[0]
        for i in tqdm(range(1, len(sequence))):
            embed = embeds[i]
            score = torch.dot(embed, current_embed)
            if score > self.config.threshold:
                continue
            current_embed = embed
            indices.append(i)
        sequence_out = sequence.clone()
        indices = torch.tensor(indices)
        sequence_out.cameras = sequence.cameras[indices]
        sequence_out.images  = sequence.images [indices]
        return sequence_out, indices


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
        cameras = sequence.transform_cameras(*renderer.get_camera_transform())
        scales  = renderer.scales

        N = len(sequence)
        M = len(scales)
        H, W = sequence.cameras[0].height, sequence.cameras[0].width

        accum_depths = []
        accum_clip_embed_means = []
        accum_clip_assignments = []
        accum_dino_embed_means = []
        accum_dino_assignments = []
        count_clip = 0
        count_dino = 0
        if self.config.visualize_dir is not None:
            pcas_clip = defaultdict(dict)
            pcas_dino = {}
            os.makedirs(self.config.visualize_dir, exist_ok=True)
        
        for i, camera in tqdm(enumerate(cameras), 'Rendering views'):
            outputs = renderer.render(camera)
            image = (outputs['rgb'] * 255).cpu().to(torch.uint8)
            accum_depths.append(outputs['depth'].cpu().squeeze(-1))
            
            for j, scale in tqdm(enumerate(scales), leave=False):
                embed_clip = renderer.render_scale(camera, scale).cpu()
                embed_mean, assignment = quantize_image_superpixel(image, embed_clip)
                accum_clip_embed_means.append(embed_mean)
                accum_clip_assignments.append(assignment + count_clip)
                count_clip += len(torch.unique(assignment))

                if self.config.visualize_dir is not None:
                    pca = compute_pca(embed_clip.numpy())
                    pcas_clip[i][j] = pca
                    embed_pred = embed_mean[assignment]
                    visualize_features(embed_clip.numpy(), pca).save(f'{self.config.visualize_dir}/clip_{i:003}_{scale:.3f}.png')
                    visualize_features(embed_pred.numpy(), pca).save(f'{self.config.visualize_dir}/clip_{i:003}_{scale:.3f}_quant.png')
 
            embed_dino = outputs['dino'].cpu()
            embed_mean, assignment = quantize_image_superpixel(image, embed_dino)
            accum_dino_embed_means.append(embed_mean)
            accum_dino_assignments.append(assignment + count_dino)
            count_dino += len(torch.unique(assignment))

            if self.config.visualize_dir is not None:
                pca = compute_pca(embed_dino.numpy())
                pcas_dino[i] = pca
                embed_pred = embed_mean[assignment]
                visualize_features(embed_dino.numpy(), pca).save(f'{self.config.visualize_dir}/dino_{i:003}.png')
                visualize_features(embed_pred.numpy(), pca).save(f'{self.config.visualize_dir}/dino_{i:003}_quant.png')

        accum_depths = torch.stack(accum_depths)
        accum_clip_embed_means = torch.cat(accum_clip_embed_means, dim=0)
        accum_dino_embed_means = torch.cat(accum_dino_embed_means, dim=0)
        accum_clip_assignments = torch.stack(accum_clip_assignments).reshape(N, M, H, W)
        accum_dino_assignments = torch.stack(accum_dino_assignments)
        # print(accum_clip_embed_means.shape)
        # print(accum_dino_embed_means.shape)
        # print(accum_clip_assignments.shape)
        # print(accum_dino_assignments.shape)

        clip_codebook, clip_codebook_indices = setup_codebook(
            accum_clip_embed_means, 
            accum_clip_assignments, 
            k=int(self.config.k_clip_ratio * len(accum_clip_embed_means))
        )
        dino_codebook, dino_codebook_indices = setup_codebook(
            accum_dino_embed_means, 
            accum_dino_assignments, 
            k=int(self.config.k_dino_ratio * len(accum_dino_embed_means))
        )
        # print(clip_codebook.shape)
        # print(dino_codebook.shape)
        # print(clip_codebook_indices.shape)
        # print(dino_codebook_indices.shape)

        if self.config.visualize_dir is not None:
            for i, j in itertools.product(range(N), range(M)):
                embed_pred = clip_codebook[clip_codebook_indices[i, j]]
                pca = pcas_clip[i][j]
                visualize_features(embed_pred.numpy(), pca).save(f'{self.config.visualize_dir}/clip_{i:003}_{scales[j]:.3f}_quant_codebook.png')
            for i in range(N):
                embed_pred = dino_codebook[dino_codebook_indices[i]]
                pca = pcas_dino[i]
                visualize_features(embed_pred.numpy(), pca).save(f'{self.config.visualize_dir}/dino_{i:003}_quant_codebook.png')
        
        sequence_out = sequence.clone()
        sequence_out.depths = accum_depths
        sequence_out.clip_codebook = clip_codebook
        sequence_out.dino_codebook = dino_codebook
        sequence_out.clip_codebook_indices = clip_codebook_indices
        sequence_out.dino_codebook_indices = dino_codebook_indices
        return sequence_out


if __name__ == '__main__':
    from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader

    reader = LERFFrameSequenceReader('/home/gtangg12/data/lerf/LERF Datasets/', 'bouquet')
    sequence = reader.read(slice=(0, None, 1))
    renderer = Renderer('/home/gtangg12/efficient-lerf/outputs/bouquet/lerf/2024-11-07_112933/config.yml')

    camera_traj_quant = CameraTrajQuantization(OmegaConf.create({'threshold': 0.4}))
    print(len(sequence))
    sequence, sequence_indices = camera_traj_quant.process_sequence(sequence)
    print(len(sequence))
    print(sequence_indices)
    
    #sequence = reader.read(slice=(0, 10, 5))
    feature_map_quant = FeatureMapQuantization(OmegaConf.create({
        'k_clip_ratio': 0.1, 
        'k_dino_ratio': 0.1, 
        'visualize_dir': reader.data_dir / 'sequence/visualizations'
    }))
    sequence = feature_map_quant.process_sequence(sequence, renderer)
    print(len(sequence))

    print(sequence.images.shape)
    print(sequence.depths.shape)
    print(sequence.cameras.shape)
    print(sequence.clip_codebook.shape)
    print(sequence.dino_codebook.shape)
    print(sequence.clip_codebook_indices.shape)
    print(sequence.dino_codebook_indices.shape)
    print(sequence.metadata)

    save_sequence(reader.data_dir / 'sequence', sequence)
    sequence2 = load_sequence(reader.data_dir / 'sequence')

    assert torch.allclose(sequence.images, sequence2.images)
    assert torch.allclose(sequence.depths, sequence2.depths)
    assert torch.allclose(sequence.cameras.camera_to_worlds, sequence2.cameras.camera_to_worlds)
    assert torch.allclose(sequence.clip_codebook, sequence2.clip_codebook)
    assert torch.allclose(sequence.dino_codebook, sequence2.dino_codebook)
    assert torch.allclose(sequence.clip_codebook_indices, sequence2.clip_codebook_indices)
    assert torch.allclose(sequence.dino_codebook_indices, sequence2.dino_codebook_indices)
    print(sequence2.metadata)