from __future__ import absolute_import

from hls4ml import converters
from hls4ml import report
from hls4ml import utils

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

def reseed(newseed):
    print('\npytest-randomly: reseed with {}'.format(newseed))
    try:
        import tensorflow
        tensorflow.random.set_seed(newseed)
    except ImportError:
        print('\nTensorFlow seed not set')
    try: 
        import torch
        torch.manual_seed(newseed)
    except ImportError:
        print('\nPyTorch seed not set')
