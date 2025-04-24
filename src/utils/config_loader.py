import yaml
from types import SimpleNamespace

def load_config(path):
    with open(path, 'r') as stream:
        raw_cfg = yaml.safe_load(stream)
    return dict_to_namespace(raw_cfg)

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d