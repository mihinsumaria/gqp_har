from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


class LogisticModel(SGDClassifier):
    def __init__(self, penalty=None, early_stopping=True, loss='log',
                 learning_rate='optimal', class_weight='balanced',
                 max_iter=10000, shuffle=True, **kwargs):
        super().__init__(penalty=penalty, early_stopping=early_stopping,
                         loss=loss, learning_rate=learning_rate,
                         class_weight=class_weight, max_iter=max_iter,
                         shuffle=shuffle, **kwargs)


class Models(Enum):
    logistic = 1
    random_forest = 2
    xgboost = 3
    knn = 4
    svc = 5

Models.logistic.model = LogisticModel
Models.random_forest.model = RandomForestClassifier
Models.xgboost.model = XGBClassifier
Models.knn.model = KNeighborsClassifier
Models.svc.model = SVC
