import sys
sys.path.append('/home/dell/Desktop/yuan_code/point-change-detection/utils')

#from utils import registry
import registry


MODELS = registry.Registry('models')

def build_model_from_cfg(cfg, **kwargs):
    return MODELS.build(cfg, **kwargs)
