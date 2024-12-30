import json
from collections import defaultdict
from pathlib import Path

from efficient_lerf.data.sequence import FrameSequence, load_sequence

from experiments.common import RENDERERS, DATASETS, summarize, setup


SAVE_DIR = Path('experiments/codebook')


def bits_per_dim(sequence: FrameSequence) -> dict:
    """
    """
    stats = defaultdict(dict)
    for feature_name in sequence.codebook_vectors.keys():
        codebook_vectors = sequence.codebook_vectors[feature_name]
        codebook_indices = sequence.codebook_indices[feature_name]
        stats[feature_name]['bits/dim'] = codebook_vectors.element_size() * codebook_vectors.nelement() * 8 / codebook_indices.nelement()
    return stats


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type) -> dict:
    """
    """
    stats, path, renderer_name = setup(SAVE_DIR, scene, RendererT)
    print(f'Evaluating codebook for renderer {renderer_name} for scene {scene}')
    if stats is not None:
        return stats
    
    reader = FrameSequenceReaderT(scene)
    sequence = load_sequence(reader.data_dir / 'sequence/sequence.pt')

    stats = bits_per_dim(sequence)
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


if __name__ == '__main__':
    accum = {}
    for RendererT, FrameSequenceReaderT in RENDERERS:
        for scene in DATASETS:
            accum[(scene, RendererT)] = evaluate(scene, RendererT, FrameSequenceReaderT)
    summarize(SAVE_DIR, accum)