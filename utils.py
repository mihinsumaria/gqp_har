import json
import pickle
import shutil

from ruamel import yaml


def terminal_break():
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    print("=" * terminal_width)


def pickle_dump(object, file):
    with open(file, 'wb') as f:
        pickle.dump(object, f)


def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def yaml_load(file):
    with open(file, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def dict_dump(d, file):
    with open(file, 'w') as f:
        f.write(json.dumps(d, indent=2))
