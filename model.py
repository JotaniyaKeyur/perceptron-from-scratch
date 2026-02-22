import numpy as np

class PerceptronScratch:

    def __init__(self, w1, w2, b, lr):
        self.w1 = w1
        self.w2 = w2
        self.b = b
        self.lr = lr

    def linear(self, x1, x2):
        return self.w1 * x1 + self.w2 * x2 + self.b

    def activation(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, y_pred, y_true):
        return - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, x1, x2, y_true, y_pred):

        dz = y_pred - y_true

        dw1 = dz * x1
        dw2 = dz * x2
        db = dz

        self.w1 -= self.lr * dw1
        self.w2 -= self.lr * dw2
        self.b -= self.lr * db
