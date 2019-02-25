import argparse
import os
import time

from sklearn.metrics import classification_report

from config import Config
from utils import terminal_break, pickle_dump, yaml_dump

TRAINING_RESULTS = './local/results'
MODELS_PATH = './local/models'
CV_PARAMS_PATH = './local/params'


def main(args):
    config_path = args.config
    name = args.name + "_{}".format(int(time.time()))
    config = Config(config_path)

    if not os.path.exists(TRAINING_RESULTS):
        os.makedirs(TRAINING_RESULTS)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    if not os.path.exists(MODELS_PATH):
        os.makedirs(MODELS_PATH)

    X_train, X_test, y_train, y_test = config.get_data_from_config()
    grid = config.get_estimator_from_config()
    grid.fit(X_train, y_train)
    terminal_break()
    print("Training finished")

    predictions = grid.predict(X_test)
    report = classification_report(y_test, predictions)
    report_path = os.path.join(TRAINING_RESULTS, name + '_report.txt')
    print("Classification Report stored in {}".format(report_path))
    print(report)

    with open(report_path, 'w') as f:
        f.write(report)
    model_path = os.path.join(MODELS_PATH, name + '_model.pkl')
    print("\n Pickling and saving best model at {}".format(model_path))
    pickle_dump(grid.best_estimator_, model_path)

    cv_params_and_score = {'best_score': grid.best_score_,
                           'best_params': grid.best_params_}
    params_path = os.path.join(CV_PARAMS_PATH, name + '_params.yml')
    yaml_dump(cv_params_and_score, params_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True,
                        help='Path to config file')
    parser.add_argument('-n', '--name', required=True,
                        help='Name of config/model')
    args = parser.parse_args()

    main(args)
