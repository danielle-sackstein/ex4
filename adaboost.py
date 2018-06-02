"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

Author: Noga Zaslavsky
Edited: Yoav Wald, May 2018

"""
import numpy
import numpy as np


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights

    def train(self, X, y):
        """
        Train this classifier over the sample (X,y)
        """
        m = X.shape[0]
        D = np.asarray([1 / m] * m)

        for t in range(self.T):
            self.h[t] = self.WL(D, X, y)
            e = self._curr_error(self.h[t], D, X, y)

            self.w[t] = 0.5 * np.math.log(1 / e - 1) if e > 0.00000001 else 100
            self._update_distribution(D, X, y, self.h[t], self.w[t])

        return self.predict(X)

    def predict(self, X):
        """
        Returns
        -------
        y_hat : a prediction vector for X
        """

        y_pred = [0]*X.shape[0]
        for t in range(self.T):
            y_pred_t = self.h[t].predict(X)
            y_pred += y_pred_t * self.w[t]

        return np.asarray([1 if s > 0 else -1 for s in y_pred])

    def error(self, X, y):
        """
        Returns
        -------
        the error of this classifier over the sample (X,y)
        """

        y_pred = self.predict(X)

        sample_count = X.shape[0]
        correct_predicts = 0
        for i in range(sample_count):
            correct_predicts += y_pred[i] * y[i]

        errors = sample_count - correct_predicts
        return errors / sample_count

    def _update_distribution(self, D, X, y, h, w):
        y_pred = h.predict(X)
        denominator = 0
        for i in range(len(X)):
            denominator += self._weight(D, w, y, y_pred, i)

        for i in range(len(X)):
            D[i] = D[i] * self._weight(D, w, y, y_pred, i) / denominator

    def _weight(self, D, w, y, y_pred, i):
        return D[i] * np.math.exp(- w * y_pred[i] * y[i])

    def _curr_error(self, h, D, X, y):
        y_hat = h.predict(X)
        total_error = 0
        for i in range(y_hat.shape[0]):
            error = 0 if y_hat[i] == y[i] else 1
            total_error += D[i] * error
        return total_error
