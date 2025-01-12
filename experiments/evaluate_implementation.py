import json
from pathlib import Path
from collections import defaultdict

from efficient_lerf.data.sequence import load_sequence, FrameSequence

from experiments.common import RENDERERS, DATASETS, summarize, setup


SAVE_DIR = Path('experiments/outputs/implementation')
MEM2UNIT = {'B': 1, 'KB': 1024, 'MB': 1024 ** 2, 'GB': 1024 ** 3}


def compute_runtime(sequence: FrameSequence) -> dict:
    """
    """
    time = sequence.metadata['quantization_duration']
    return {'global': {'time': time / len(sequence)}} # all features combined


def compute_memory(sequence: FrameSequence, unit='MB') -> dict:
    """
    """
    memory = defaultdict(dict)
    for feature_name in sequence.codebook_vectors.keys():
        codebook_vectors = sequence.codebook_vectors[feature_name]
        codebook_indices = sequence.codebook_indices[feature_name]
        mem_codebook = codebook_vectors.element_size() * codebook_vectors.nelement() / MEM2UNIT[unit]
        mem_cindices = codebook_indices.element_size() * codebook_indices.nelement() / MEM2UNIT[unit]
        mem_combined = mem_codebook + mem_cindices
        memory[feature_name] = {
            'codebook': mem_codebook / len(sequence), 
            'cindices': mem_cindices / len(sequence),
            'combined': mem_combined / len(sequence)
        }
    return memory


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type) -> dict:
    """
    """
    stats, path, renderer_name = setup(SAVE_DIR, scene, RendererT)
    print(f'Evaluating codebook for renderer {renderer_name} for scene {scene}')
    if stats is not None:
        return stats

    reader = FrameSequenceReaderT(scene)
    sequence = load_sequence(reader.data_dir / 'sequence/sequence.pt')
    stats = {**compute_runtime(sequence), **compute_memory(sequence)}
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


if __name__ == '__main__':
    accum = {}
    for RendererT, FrameSequenceReaderT in RENDERERS:
        for scene in DATASETS:
            accum[(scene, RendererT)] = evaluate(scene, RendererT, FrameSequenceReaderT)
    summarize(SAVE_DIR, accum)