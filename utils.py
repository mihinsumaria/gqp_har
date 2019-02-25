import shutil
import pickle
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


def yaml_dump(data, file):
    with open(file, 'r') as f:
        yaml.safe_dump(data, f)


def yaml_load(file):
    with open(file, 'r') as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml
