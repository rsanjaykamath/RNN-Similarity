import os
from ..tokenizers import SpacyTokenizer
from .. import DATA_DIR


DEFAULTS = {
    'tokenizer': SpacyTokenizer,
}


def set_default(key, value):
    global DEFAULTS
    DEFAULTS[key] = value

from .model import RnnSimilarModel
from .predictor import Predictor
from . import config
from . import vector
from . import data
from . import utils
