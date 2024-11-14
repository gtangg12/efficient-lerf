from efficient_lerf.data.common import DATASETS, DATASET_DIR
from efficient_lerf.data.sequence import load_sequence, FrameSequence


def memory_usage(sequence: FrameSequence):
    """
    """
    memory = 0
    memory += sequence.images.data.element_size() * sequence.images.data.nelement()
    memory += sequence.depths.data.element_size() * sequence.depths.data.nelement()
    memory += sequence.clip_codebook_indices.data.element_size() * sequence.clip_codebook_indices.data.nelement()
    memory += sequence.dino_codebook_indices.data.element_size() * sequence.dino_codebook_indices.data.nelement()
    memory += sequence.clip_codebook.data.element_size() * sequence.clip_codebook.data.nelement()
    memory += sequence.dino_codebook.data.element_size() * sequence.dino_codebook.data.nelement()
    return memory


if __name__ == '__main__':
    sum = 0
    for scene in DATASETS:
        sequence = load_sequence(DATASET_DIR / scene / scene / 'sequence')

        memory = memory_usage(sequence)
        N = len(sequence)
        npoints = N * sequence.cameras[0].height.item() * sequence.cameras[0].width.item() # duality
        print(f' Sequence len: {N}')
        print(f' Memory: {memory / 1e9:.2f} GB')
        print(f' Memory per point: {memory / npoints:.2f} bytes')
        sum += memory / npoints
    print(f' Average memory per point: {sum / len(DATASETS):.2f} bytes')