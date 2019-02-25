# Human Activity Recognition
The objective of this project is to explore and evaluate application of Machine Learning/Deep Learning algorithms to solve problem of Human Activity Recognition using labelled public actigraphy data.

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites
Create a virtual environment for python3 and install using the required python packages using the following command:

`pip install -r requirements.txt`

### Dataset
Please download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones), and unzip it in the `data` directory.

### Running the Project
Please find example configs in the `configs` directory. After creating and adding your config to the directory, you can run the `train.py`. You will have to provide the path to the config file via command line, and also give a name to the config/model. You can find more information for the command line arguments by running.

`train.py --help`

For eg.

`train.py --config ./configs/pca_logistic_config.yml --name pca_logistic`

### Results
The test results will be stored in `local/results` and the trained model will be pickled and stored in `local/models`.


