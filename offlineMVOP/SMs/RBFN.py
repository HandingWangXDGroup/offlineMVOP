# Radial basis function network

import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator


class RBFN(BaseEstimator):

    def __init__(self, num_neurons = None):
        # 模型超参数
        self.num_neurons = num_neurons
        self.sigma = None
        self.centers = None

        # 参数
        self.weights = None
        self.bias = None

    def kernel_(self, data_point):  # Gaussian function
        distMat = np.sum(data_point ** 2, 1).reshape(-1, 1) + np.sum(self.centers ** 2, 1) - 2 * data_point.dot(
            self.centers.T)
        return np.exp(-0.5 * distMat / self.sigma ** 2)

    def calsigma(self):
        max = 0.0
        num = 0
        total = 0.0
        for i in range(self.num_neurons - 1):
            for j in range(i + 1, self.num_neurons):
                dis = np.linalg.norm(self.centers[i] - self.centers[j])
                total = total + dis
                num += 1
                if dis > max:
                    max = dis
        self.sigma = 2 * total / num

    def fit(self, X, Y):
        if self.num_neurons is None:
            self.num_neurons = int(np.sqrt(X.shape[0])) #int(np.sqrt(X.shape[0])) int(X.shape[0]/2)
        km = KMeans(n_clusters=self.num_neurons).fit(X)
        self.centers = km.cluster_centers_
        self.calsigma()
        G = self.kernel_(X)
        temp = np.column_stack((G, np.ones((X.shape[0]))))
        temp = np.dot(np.linalg.pinv(temp), Y)
        self.weights = temp[:self.num_neurons]
        self.bias = temp[self.num_neurons]

    def predict(self, X):
        X = np.array(X)
        G = self.kernel_(X)
        predictions = np.dot(G, self.weights) + self.bias
        return predictions

    def score(self, X, y):
        MSE = np.sum((y - self.predict(X)) ** 2) / len(y)
        R2 = 1 - MSE / np.var(y)
        return R2
