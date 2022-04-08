import numpy as np


class MLPerceptron:

    def __init__(self, neurons_layer, learning_rate=0.01, precision=0.000001, beta=0.5, alfa=0.9):
        self.learning_rate = learning_rate
        self.precision = precision
        self.beta = beta
        self.alfa = alfa
        self.neurons_layer = neurons_layer
        self.layers_outputs = np.zeros(len(self.neurons_layer)).tolist()
        self.w_sums = np.zeros(len(self.neurons_layer)).tolist()
        self.weights = np.zeros(len(self.neurons_layer)).tolist()
        self.gradients = np.zeros(len(self.neurons_layer)).tolist()

    def train(self, inputs, expected):
        prev_msq_error = 99999999
        cur_msq_error = 0
        self.build(inputs)
        while(np.absolute(cur_msq_error - prev_msq_error) > self.precision):
            prev_msq_error = cur_msq_error
            for x, y in zip(inputs, expected):
                self.compute_layers_result(x)
                self.update_weights(y)

    def build(self, inputs):
        input_length = len(inputs[0])
        layers = len(self.neurons_layer)
        self.weights[0] = np.random.rand(
            self.neurons_layer[0], input_length + 1)
        for i in range(1, layers):
            self.weights[i] = np.random.rand(
                self.neurons_layer[i], self.neurons_layer[i - 1] + 1)

    def compute_layers_result(self, line):
        layers = len(self.neurons_layer)
        line.append(-1)
        self.w_sums[0] = np.matmul(self.weights[0], line)
        self.layers_outputs[0] = self.sigmoid(self.w_sums[0])
        for i in range(1, layers):
            self.layers_outputs[i - 1] = np.append(self.layers_outputs[i - 1], -1)
            self.w_sums[i] = np.matmul(self.weights[i], self.layers_outputs[i - 1])
            self.layers_outputs[i] = self.sigmoid(self.w_sums[i])

    def sigmoid(self, array):
        def sigmoid(x): return 1/(1 + np.exp(-x))
        return np.array([sigmoid(x) for x in array])

    def sigmoid_derivative(self, array):
        def sigmoid_derivative(x): return self.beta * \
            (1/(1 + np.exp(-x))) * (1 - (1/(1 + np.exp(-x))))
        return np.array([sigmoid_derivative(x) for x in array])

    def update_weights(self, expected):
        for layer in reversed(range(len(self.neurons_layer))):
            if(layer == len(self.neurons_layer) - 1):
                derivative = self.sigmoid_derivative(self.w_sums[layer])
                self.gradients[layer] = (np.subtract(np.array(expected), np.array(self.layers_outputs[layer]))) * derivative
                new_gradient = []
                for g in self.gradients[layer]:
                    new_gradient.append([g])
                self.gradients[layer] = new_gradient
                gradient_lr = np.multiply(self.gradients[layer], self.learning_rate)
                update_matrix = np.multiply(gradient_lr, self.layers_outputs[layer - 1])
                print(self.weights[layer], 'before')
                self.weights[layer] = np.add(self.weights[layer], update_matrix)
                print(self.weights[layer], 'after')
            elif(layer == 0):
                print('Needs to be implemented')
            else:
                derivative = self.sigmoid_derivative(self.w_sums[layer])
                gradient_factor = np.multiply(self.gradients[layer + 1], self.weights[layer + 1])
                self.gradients[layer] = gradient_factor * derivative
                new_gradient = []
                for g in self.gradients[layer]:
                    new_gradient.append([g])
                self.gradients[layer] = new_gradient
                gradient_lr = np.multiply(self.gradients[layer], self.learning_rate)
                update_matrix = np.multiply(self.layers_outputs[layer - 1], gradient_lr)
                self.weights[layer] = np.add(self.weights[layer], update_matrix)
 