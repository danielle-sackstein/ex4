"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Running script for Ex4.

Author:
Date: May, 2018

"""

import numpy as np

from adaboost import AdaBoost
from ex4_tools import decision_boundaries, h_opt, DecisionStump
from matplotlib.pyplot import *


def Q3():  # AdaBoost
    # TODO - implement this function
    return


def Q4():  # decision trees
    # TODO - implement this function
    return


def Q5():  # spam data
    # TODO - implement this function
    return


def to_ndarray(list_):
    return np.array(list_).astype(np.float64)


def read_all_data(file_name):
    with open(file_name) as file_:
        lines = file_.readlines()

        data = []
        labels = []

        for line in lines:
            values = line.split()
            data.append(np.array(values[0:-1]))
            labels.append(values[-1])

    return to_ndarray(data), to_ndarray(labels)


def get_data(test_size):
    all_data, all_labels = read_all_data('spam.data')


def _get_file_path(name):
    return 'ex4files/SynData/{}.txt'.format(name)


def _load_data(name):
    return np.loadtxt(_get_file_path('X_' + name)), np.loadtxt(_get_file_path('y_' + name))


if __name__ == '__main__':
    X_train, y_train = _load_data('train')
    X_val, y_val = _load_data('val')

    T_values = range(5, 200, 5)
    validation_error = []
    training_error = []

    for t in T_values:
        ada_boost = AdaBoost(DecisionStump, t)
        ada_boost.train(X_train, y_train)
        validation_error.append(ada_boost.error(X_val, y_val))
        training_error.append(ada_boost.error(X_train, y_train))

    training_error_plot, = plot(T_values, training_error, linestyle='--', label='training_error')
    validation_error_plot, = plot(T_values, validation_error, linestyle='-', label='validation_error')

    legend(handles=[training_error_plot, validation_error_plot])

    title('training and validation error vs T values')
    xlabel('T values')
    ylabel('training and validation error')
    savefig('adaboost.png')
    clf()

    # TODO - run your code for questions 3-5
    pass
