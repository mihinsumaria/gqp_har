from enum import Enum

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


class Models(Enum):
    logistic = 1
    random_forest = 2
    xgboost = 3
    knn = 4
    svc = 5

Models.logistic.model = SGDClassifier
Models.random_forest.model = RandomForestClassifier
Models.xgboost.model = XGBClassifier
Models.knn.model = KNeighborsClassifier
Models.svc.model = SVC
