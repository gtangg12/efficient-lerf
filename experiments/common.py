import json
from collections import defaultdict
from glob import glob
from pathlib import Path
from natsort import natsorted

from efficient_lerf.data.common import DATASET_DIR
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader, LangSplatFrameSequenceReader
from efficient_lerf.renderer.renderer_lerf import LERFRenderer
from efficient_lerf.renderer.renderer_langsplat import LangSplatRenderer
from efficient_lerf.utils.math import mean


RENDERERS          = [(LERFRenderer, LERFFrameSequenceReader), (LangSplatRenderer, LangSplatFrameSequenceReader)]
RENDERERS_EXPANDED = [(LERFRenderer, LERFFrameSequenceReader)]


DATASETS          = ['figurines', 'ramen', 'teatime', 'waldo_kitchen']
DATASETS_EXPANDED = natsorted([Path(x).stem for x in glob(str(DATASET_DIR / 'lerf/LERF Datasets/*'))])


def convert_dict_tuple2nested(d):
    r = {}
    for k, v in d.items():
        c = r
        for x in k[:-1]:
            if x not in c:
                c[x] = {}
            c = c[x]
        c[k[-1]] = v
    return r


def summarize(path: Path | str, accum: dict, reduce_fn=mean) -> dict:
    """
    Average metrics over all scenes.
    """
    summary = defaultdict(list)
    for (scene, RendererT), stats in accum.items():
        for feature_name, metrics in stats.items():
            for k, v in metrics.items():
                summary[(feature_name, RendererT.__name__, k)].append(v)
    for k, v in summary.items():
        summary[k] = reduce_fn(v)
    summary = convert_dict_tuple2nested(summary)
    with open(path / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    return summary


def setup(path: Path | str, scene: str, RendererT: type) -> tuple:
    """
    Create output directory and load stats if available.
    """
    renderer_name = RendererT.__name__
    path = Path(path) / renderer_name / scene
    filename = path / 'stats.json'
    if filename.exists():
        with open(filename, 'r') as f:
            return json.load(f), path, renderer_name
    path.mkdir(parents=True, exist_ok=True)
    return None, path, renderer_name 