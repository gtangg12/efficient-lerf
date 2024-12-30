import json
from collections import defaultdict
from pathlib import Path

from efficient_lerf.data.sequence import FrameSequence, load_sequence
from efficient_lerf.utils.math import mean

from experiments.common import RENDERERS, DATASETS, convert_dict_tuple2nested


SAVE_DIR = Path('experiments/codebook')


def bits_per_dim(sequence: FrameSequence) -> dict:
    """
    """
    stats = {}
    for feature_name in sequence.codebook_vectors.keys():
        codebook_vectors = sequence.codebook_vectors[feature_name]
        codebook_indices = sequence.codebook_indices[feature_name]
        stats[feature_name] = codebook_vectors.element_size() * codebook_vectors.nelement() * 8 / codebook_indices.nelement()
    return stats


def evaluate(scene: str, RendererT: type, FrameSequenceReaderT: type) -> dict:
    """
    """
    renderer_name = RendererT.__name__
    print(f'Evaluating codebook for renderer {renderer_name} for scene {scene}')

    path = Path(f'{SAVE_DIR}/{renderer_name}/{scene}')
    if path.exists():
        with open(path / 'stats.json', 'r') as f:
            return json.load(f)
    path.mkdir(parents=True, exist_ok=True)
    
    reader = FrameSequenceReaderT(scene)
    sequence = load_sequence(reader.data_dir / 'sequence/sequence.pt')

    stats = bits_per_dim(sequence)
    with open(path / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    return stats


def summarize(accum: dict) -> dict:
    """
    """
    summary = defaultdict(list)
    for (scene, RendererT), stats in accum.items():
        for feature_name, value in stats.items():
            summary[(feature_name, RendererT.__name__)].append(value)
    for k, v in summary.items():
        summary[k] = mean(v)
    summary = convert_dict_tuple2nested(summary)
    with open(SAVE_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    return summary


if __name__ == '__main__':
    accum = {}
    for RendererT, FrameSequenceReaderT in RENDERERS:
        for scene in DATASETS:
            accum[(scene, RendererT)] = evaluate(scene, RendererT, FrameSequenceReaderT)
    summarize(accum)