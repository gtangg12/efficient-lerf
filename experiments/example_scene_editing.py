import sys
sys.path.append('/home/gtangg12/efficient-lerf/')
from omegaconf import OmegaConf

from efficient_lerf.data.common import CONFIGS_DIR
from efficient_lerf.feature_field import load_model
from efficient_lerf.renderer.renderer_lerf import LERFRenderer
from efficient_lerf.data.sequence_reader import LERFFrameSequenceReader
from efficient_lerf.utils.visualization import *


config = OmegaConf.load(CONFIGS_DIR / 'template.yaml')

model = load_model('teatime', config, LERFRenderer, LERFFrameSequenceReader) # auto quantize
model.edit(
    method='edit',
    positive='red apple',
    prompt='turn the apple into a green apple',
    threshold=0.6,
    save=True,
    debug=True
)

model = load_model('bouquet', config, LERFRenderer, LERFFrameSequenceReader) # auto quantize
model.edit(
    method='edit',
    positive='flowers',
    prompt='change the flowers to blue',
    threshold=0.6,
    save=True,
    debug=True
)

model = load_model('donuts', config, LERFRenderer, LERFFrameSequenceReader) # auto quantize
model.edit(
    method='edit',
    positive='donuts',
    prompt='replace the glazed donuts with chocolate donuts',
    threshold=0.65,
    save=True,
    debug=True
)

model = load_model('ramen', config, LERFRenderer, LERFFrameSequenceReader) # auto quantize
model.edit(
    method='edit',
    positive='eggs',
    prompt='make the egg yolk more runny',
    threshold=0.6,
    save=True,
    debug=True
)

model = load_model('fruit_aisle', config, LERFRenderer, LERFFrameSequenceReader) # auto quantize
model.edit(
    method='edit',
    positive='fruit',
    prompt='turn into a Monet painting',
    threshold=0.6,
    save=True,
    debug=True
)