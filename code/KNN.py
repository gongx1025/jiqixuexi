# -*- coding: utf-8 -*-
# @Author: GongXu

import os
from common import tt_split, acc_recall_prec_score, confusion_paint
from sklearn.neighbors import KNeighborsClassifier


def knn():
    x_train, x_test, y_train, y_test = tt_split()
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(x_train, y_train)
    prediction = clf.predict(x_test)
    acc_recall_prec_score(y_test, prediction)

    title = os.path.split(__file__)[-1].split(".")[0] + " Confusion Matrix"
    confusion_paint(y_test, prediction, title)


if __name__ == '__main__':
    knn()