from glob import glob
from pathlib import Path
from natsort import natsorted

from efficient_lerf.data.common import DATASET_DIR
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader, LangSplatFrameSequenceReader
from efficient_lerf.renderer.renderer_lerf import LERFRenderer
from efficient_lerf.renderer.renderer_langsplat import LangSplatRenderer


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