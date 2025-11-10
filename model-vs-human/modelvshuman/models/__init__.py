from .pytorch import model_zoo as pytorch_model_zoo
# TensorFlow models not needed for shape-bias plotting
try:
    from .tensorflow import model_zoo as tensorflow_model_zoo
except ImportError:
    tensorflow_model_zoo = None
from .registry import list_models

