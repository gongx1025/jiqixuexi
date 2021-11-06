# -*- coding: utf-8 -*-
# @Author: GongXu

import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 读取数据集
def get_data():
    data = pd.read_csv("../dataset/penguins_size.csv")
    data = data[['species', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
    return data


# 缺失值填充
def data_fill():
    data = get_data()
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna(data.iloc[:, 1:].median())
    return data


# species列转化为分类数值
def trans():
    data = data_fill()
    y = data.loc[:, 'species']
    le = LabelEncoder()
    label = le.fit_transform(y)
    data.loc[:, 'species'] = label
    return data


# 分割数据集 test_size = 0.3, random_state = 22
def tt_split(test_size=0.3, random_state=22):
    data = trans()
    x_train, x_test, y_train, y_test = train_test_split(
        data[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']],
        data[['species']],
        test_size=test_size,
        random_state=random_state)
    return x_train, x_test, y_train, y_test


# 标准化
def standard(x_train, x_test):
    std = StandardScaler()
    x_train = std.fit_transform(x_train)
    x_test = std.fit_transform(x_test)
    return x_train, x_test


# 模型评估指标
def acc_recall_prec_score(y_true, prediction, average='macro'):
    print("准确率：", accuracy_score(y_true, prediction))
    print("召回率：", recall_score(y_true, prediction, average=average))
    print("精确率：", precision_score(y_true, prediction, average=average))


# 绘制混淆矩阵
def confusion_paint(y_test, prediction, title):
    classes = list(i for i in range(0, 3))
    classes.sort()
    confusion = confusion_matrix(y_test, prediction)
    plt.imshow(confusion, cmap=plt.cm.Blues)
    indices = range(len(confusion))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.title(title)

    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])
    plt.show()


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = tt_split()
    print(len(x_train))
    print(len(x_test))
