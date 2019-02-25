from pprint import pprint

import numpy as np
from ruamel import yaml
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from export import (get_data_xy, get_feature_indices, get_features,
                    get_labels_dictionary)
from models import Models
from transformers import Transformers
from utils import terminal_break


class Config:

    def __init__(self, config_file):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
            print("Config file loaded successfully: {}".format(config_file))
            terminal_break()
            pprint(self.config)
            terminal_break()

    def get_data_from_config(self):
        print("Fetching data from config")
        X_train = get_data_xy(self.data['X_train'])
        y_train = get_data_xy(self.data['y_train'])
        X_test = get_data_xy(self.data['X_test'])
        y_test = get_data_xy(self.data['y_test'])
        activity_labels = get_labels_dictionary(self.data['activity_labels'])
        y_train = np.array([activity_labels[y] for y in y_train])
        y_test = np.array([activity_labels[y] for y in y_test])
        return X_train, X_test, y_train, y_test

    def get_estimator_from_config(self):
        print("Putting together an estimator based on the config")
        steps = []
        grid_params = {}
        for transformer in self.pipeline['transformer'].keys():
            if transformer == 'subset':
                indices = get_feature_indices(self.pipeline['transformer'][
                                                transformer])
                transformer_to_append = Transformers[transformer].transformer(
                                                                    indices)
                continue
            else:
                transformer_to_append = Transformers[transformer].transformer()
            steps.append((transformer, transformer_to_append))
            for param, value in self.pipeline['transformer'][
                    transformer].items():
                grid_params[transformer + "__" + param] = value

        classifier_name = list(self.pipeline['classifier'].keys())[0]
        classifier = Models[classifier_name].model
        for param, value in self.pipeline['classifier'][
                classifier_name].items():
            grid_params[classifier_name + "__" + param] = value
        steps.append((classifier_name, classifier()))
        pipe = Pipeline(steps=steps)
        cv = self.pipeline['grid']['cv']
        scoring = self.pipeline['grid']['scoring']
        grid = GridSearchCV(pipe, grid_params, scoring=scoring, cv=cv,
                            verbose=1)
        return grid

    def __getattr__(self, item):
        return self.config.get(item)
