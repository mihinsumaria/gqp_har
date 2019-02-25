import shutil
import pickle


def terminal_break():
    terminal_width = shutil.get_terminal_size((80, 20)).columns
    print("=" * terminal_width)


def pickle_dump(object, file):
    with open(file, 'wb') as f:
        pickle.dump(object, f)


def pickle_load(file):
    with open(file, 'rb') as f:
        return pickle.load(f)
