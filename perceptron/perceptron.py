import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def train(self, X):
        [self.weights] = np.random.rand(1, len(X[0]) - 1)
        self.weights = np.array(self.weights)
        print('Initial Weights:', self.weights)
        epochs = 1
        while True:
            error = False
            for x in X:
                inputs = np.array(x[0:len(x) - 1])
                expected_y = x[len(x) - 1]
                calculated_y = self.signal(inputs)
                if(expected_y != calculated_y):
                    error = True
                    update = self.learning_rate * (expected_y - calculated_y)
                    update_array = np.multiply(inputs, update)
                    self.weights = np.add(update_array, self.weights)
            if not error:
                break
            epochs += 1
        print('Final Weights:', self.weights)
        print('Epochs:', epochs)

    def test(self, inputs):
        return self.signal(inputs)

    def signal(self, X):
        inner_p = np.inner(X, self.weights)
        if inner_p >= 0:
            return 1
        else:
            return -1
