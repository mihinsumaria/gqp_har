import os

import numpy as np
import pandas as pd


def get_features(features_path):
    """
    Fetch features for the HAR dataset

    Args:
        features_path (str): Path to the feature names

    Returns:
        features (list): List containing all feature names
    """
    with open(features_path, 'r') as file:
        features = file.readlines()
    features = [feature.strip() for feature in features]
    return features


def get_feature_indices(subset_path, features_path='./data/features.txt'):
    """
    Get feature indices from overall feature set for a subset for
    FeatureSubsetTransformer

    Args:
        subset_path (str): path to subset features
        features_path (str): path to all features
    """
    subset = get_features(subset_path)
    features = get_features(features_path)
    indices = [features.index(feature) for feature in subset]
    return indices


def get_labels_dictionary(labels_path):
    """
    Fetch labels for the HAR dataset

    Args:
        labels_path (str): Path to the file containing label dict

    Returns: (dict) Label dictionary containing label-code and label pairs

    """
    with open(labels_path, 'r') as f:
        return {int(line.split()[0]): line.split()[1] for line in f}


def get_data_xy(path):
    """
    Fetch X and Y data for the HAR dataset as numpy arrays
    Data has to have same number of columns for each row, should be only
    numeric

    Args:
        path (str): Path to data to be imported

    Returns:
        data (numpy.array): Data imported from given path
    """
    data = np.loadtxt(path)
    return data


def get_df(id_path, x_path, y_path, features_path, labels_path):
    """
    Combine x, y, features, and IDs for the HAR dataset

    Args:
        id_path (str): Path to unique patient identifiers
        x_path (str): Path to x (data without labels and feature names)
        y_path (str): Path to y (label codes)
        features_path(str): Path to feature names
        labels_path (str): Path to file containing label dict

    Returns:
        df (pd.DataFrame): dataframe containing the fetched HAR data
        """
    id_df = pd.read_csv(id_path, header=None, names=['id'])
    features = get_features(features_path)
    x = pd.DataFrame(get_data_xy(x_path), columns=features)
    y = pd.DataFrame(get_data_xy(y_path), columns=['activity'])
    y.replace(to_replace=get_labels_dictionary(labels_path), inplace=True)
    df = pd.concat([id_df, x, y], axis=1)
    return df
