from re import I
import numpy as np


class MLPerceptron:

    def __init__(self, neurons_layer, learning_rate=0.01, precision=0.000001, beta=0.5, alfa=0.9):
        self.learning_rate = learning_rate
        self.precision = precision
        self.beta = beta
        self.alfa = alfa
        # Array containing the toplogy of the network
        self.neurons_layer = neurons_layer
        # Initialize layers as a zeroes array
        self.layers_outputs = np.zeros(len(self.neurons_layer)).tolist()
        self.w_sums = np.zeros(len(self.neurons_layer)).tolist()
        self.weights = np.zeros(len(self.neurons_layer)).tolist()
        self.gradients = np.zeros(len(self.neurons_layer)).tolist()
        self.prev_msq_error = 99999999
        self.cur_msq_error = 0

    def fit(self, inputs, expected):
        # Initialize error
        self.prev_msq_error = 99999999
        self.cur_msq_error = 0
        # Build the network
        self.build(inputs)
        print(self.weights)
        while(np.absolute(self.cur_msq_error - self.prev_msq_error) > self.precision):
            self.prev_msq_error = self.cur_msq_error
            msq_error = 0
            for x, y in zip(inputs, expected):
                self.compute_layers_result(x)
                self.update_weights(y, x)
                msq_error += self.calculate_error(y)
            self.cur_msq_error = msq_error / len(inputs)
        print(self.weights)

    def test(self, inputs, expected):
        for x, y in zip(inputs, expected):
            self.compute_layers_result(x, True)
            print('result', self.layers_outputs[-1], y)

    def build(self, inputs):
        input_length = len(inputs[0])
        layers = len(self.neurons_layer)
        # Add weight matrix to the first position using the length of the input plus the bias weight
        self.weights[0] = np.random.rand(
            self.neurons_layer[0], input_length + 1)
        for i in range(1, layers):
            # Add weight matrix to the subsequent positions using the previous layers outputs plus the bias weight
            self.weights[i] = np.random.rand(
                self.neurons_layer[i], self.neurons_layer[i - 1] + 1)

    def compute_layers_result(self, line, isTest=False):
        layers = len(self.neurons_layer)
        # Add bias to the input layer
        if(self.cur_msq_error == 0 or isTest):
            line.append(-1)
        # Multiply the weight matrix by the inputs
        self.w_sums[0] = np.matmul(self.weights[0], line)
        # Multiply resulting array by the activation function
        self.layers_outputs[0] = self.sigmoid(self.w_sums[0])
        for i in range(1, layers):
            # Add bias to the last layer output
            self.layers_outputs[i -
                                1] = np.append(self.layers_outputs[i - 1], -1)
            # Multiply the weight matrix by the last layer output
            self.w_sums[i] = np.matmul(
                self.weights[i], self.layers_outputs[i - 1])
            # Multiply resulting array by the activation function
            self.layers_outputs[i] = self.sigmoid(self.w_sums[i])

    # Apply sigmoid function to the entire array
    def sigmoid(self, array):
        def sigmoid(x): return 1/(1 + np.exp(-self.beta * x))
        return np.array([sigmoid(x) for x in array])

    # Apply sigmoid derivative function to the entire array
    def sigmoid_derivative(self, array):
        def sigmoid_derivative(x): return self.beta * (
            1/(1 + np.exp(-self.beta * x))) * (1 - (1/(1 + np.exp(-self.beta * x))))
        return np.array([sigmoid_derivative(x) for x in array])

    def update_weights(self, expected, inputs):
        # Iterate over the reversed topology to apply backpropagation algorithm
        for layer in reversed(range(len(self.neurons_layer))):
            if(layer == len(self.neurons_layer) - 1):
                # Calculate gradient for the last layer
                derivative = self.sigmoid_derivative(self.w_sums[layer])
                difference = np.subtract(np.array(expected), np.array(self.layers_outputs[layer]))
                # (dj - Y(i)) * g´(I(i))
                gradients = np.multiply(difference, derivative)
                self.gradients[layer] = self.format_gradients(gradients)
                gradient_lr = np.multiply(
                    self.gradients[layer], self.learning_rate)
                update_matrix = np.multiply(
                    gradient_lr, self.layers_outputs[layer - 1])  # n * δ * Y(i - 1)
                self.weights[layer] = np.add(
                    self.weights[layer], update_matrix)
            else:
                # Calculate gradient for the previous layers
                derivative = self.sigmoid_derivative(self.w_sums[layer])
                gradient_factor = np.multiply(self.gradients[layer + 1], self.weights[layer + 1])
                gradients = np.multiply(self.gradient_by_neuron(gradient_factor), derivative)
                self.gradients[layer] = self.format_gradients(gradients)
                gradient_lr = np.multiply(self.gradients[layer], self.learning_rate)
                update_matrix = []
                if(layer == 0):
                    update_matrix = np.multiply(inputs, gradient_lr)
                else:
                    update_matrix = np.multiply(self.layers_outputs[layer - 1], gradient_lr)
                self.weights[layer] = np.add(self.weights[layer], update_matrix)

    def format_gradients(self, gradients):
        # Format gradients to be able to multiply them by the layer outputs
        new_gradients = []
        for g in gradients:
            new_gradients.append([g])
        return new_gradients

    def gradient_by_neuron(self, gradient_factor):
        # Sum the gradient by weight of the next linked neuron
        gradient_by_neuron = np.sum(gradient_factor, axis=0).tolist()
        # Return a list without the bias
        return gradient_by_neuron[:-1]

    def calculate_error(self, expected):
        difference = np.sum(np.subtract(np.array(expected),
                            np.array(self.layers_outputs[-1])))
        return np.power(difference, 2)/2
