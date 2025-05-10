# ==== Standard Imports ====
from types import SimpleNamespace

# ==== Third Party Imports
import yaml


def load_config(path):
    """
    Loads a YAML configuration file and converts it into a nested
    SimpleNamespace object.

    Args:
    :param path: The path to the YAML configuration file.
    :return: A namespace object.
    """
    with open(path, 'r') as stream:
        raw_cfg = yaml.safe_load(stream)
    return _dict_to_namespace(raw_cfg)

def _dict_to_namespace(d):
    """
    Recursively converts a dictionary (or list of dictionaries)
    into a SimpleNamespace.

    Args:
    :param d: The dictionary of list of dictionaries to convert.
    :return: SimpleNamespace object or a list namespace objects.
    """
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d