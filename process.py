import csv
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def read_data(test_data, n=1, label=1):
    csv_reader = csv.reader(open(test_data))
    data_list = []
    for one_line in csv_reader:
        data_list.append(one_line)
    x_list = []
    y_list = []
    for one_line in data_list[1:]:
        if label == 1:
            y_list.append(int(one_line[-1]))
            one_list = [float(o) for o in one_line[n:-1]]
            x_list.append(one_list)
        else:
            one_list = [float(o) for o in one_line[n:]]
            x_list.append(one_list)
    return np.mat(x_list), np.mat(y_list)


def split_data(data_list, y_list, ratio):

    x_train, x_test, y_train, y_test = train_test_split(data_list, y_list, test_size=ratio)
    return x_train, x_test, y_train, y_test


def write_data(data, csv_name):
    df = pd.DataFrame(data, columns=['id', 'categories'])
    df.to_csv(csv_name, index=False)
