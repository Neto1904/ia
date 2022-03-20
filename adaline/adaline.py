import numpy as np
import matplotlib.pyplot as charts
import uuid


class Adaline:
    def __init__(self, learning_rate=0.0025, precision_rate=0.000001):
        self.learning_rate = learning_rate
        self.precision_rate = precision_rate

    def train(self, X):
        input_array_size = len(X[0]) - 1
        [self.weights] = np.random.rand(1, input_array_size)
        msq_error_prev = 9999999999
        msq_error_current = 1
        print('Initial weigths:', self.weights)
        epochs = 0
        epochs_array = []
        errors_array = []
        while(np.absolute(msq_error_current - msq_error_prev) > self.precision_rate):
            epochs += 1
            epochs_array.append(epochs)
            msq_error_prev = msq_error_current
            msq_error = 0
            for x in X:
                inputs = x[0: len(x) - 1]
                expected = x[len(x) - 1]
                inputs_product = np.inner(inputs, self.weights)
                difference = expected - inputs_product
                update = self.learning_rate * (difference)
                update_array = np.multiply(inputs, update)
                self.weights = np.add(self.weights, update_array)
                msq_error += np.power(difference, 2)
            msq_error_current = msq_error / len(X)
            errors_array.append(msq_error_current)
        print('Final weigths:',  self.weights)
        print('Epochs:',  epochs)
        self.plot(epochs_array, errors_array, 'Epochs', 'Error',
                  'Square mean error by epochs of training')

    def test(self, x):
        result = np.inner(x, self.weights)
        if(result >= 0):
            return 'B'
        else:
            return 'A'

    def plot(self, x, y, xlabel, ylabel, title):
        charts.plot(x, y)
        charts.xlabel(xlabel)
        charts.ylabel(ylabel)
        charts.title(title)
        charts.savefig(f'./charts/{str(uuid.uuid4())}.png')
