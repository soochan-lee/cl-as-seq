from .model import Model
from .class_incremental_model import ClassIncrementalModel
from .continual_transformer import ContinualTransformer
from .oml import OML
from .anml import ANML

MODEL = {
    'ClassIncrementalModel': ClassIncrementalModel,
    'ContinualTransformer': ContinualTransformer,
    'OML': OML,
    'ANML': ANML,
}
