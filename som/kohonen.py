import numpy as np
import seaborn as sns
import matplotlib.pyplot as charts
import pandas as pd


class Kohonen:

    def __init__(self, width, length, learning_rate=0.01, radius=1):
        self.learning_rate = learning_rate
        self.width = width
        self.length = length
        self.radius = radius
        self.grid = []
        self.winners = []

    def fit(self, inputs):
        self.build(len(inputs[0]), len(inputs))
        winner_changed = True
        epochs = 0
        while(winner_changed):
            winner_changed = False
            epochs += 1
            for i in range(len(inputs)):
                [line, column] = self.calculate_dist(inputs[i])
                [last_line, last_column] = self.winners[i]
                if(last_line - line != 0 or last_column - column != 0):
                    self.winners[i] = [line, column]
                    winner_changed = True
                self.update_weights(line, column, inputs[i])
        print('Epochs: ', epochs)

    def build(self, input_length, list_length):
        print('Layout:', self.width, self.length)
        self.grid = np.zeros((self.width, self.length, input_length))
        self.winners = np.zeros((list_length, 2)).tolist()
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                self.grid[i, j] = np.random.rand(1, input_length)

    def calculate_dist(self, x):
        distance = 999999
        line = -1
        column = -1
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                new_dist = np.linalg.norm(x - self.grid[i, j])
                if(new_dist < distance):
                    distance = new_dist
                    line = i
                    column = j
        return [line, column]

    def update_weights(self, line, column, x):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                if(i == line and j == column):
                    self.grid[i, j] = self.grid[i, j] + \
                        self.learning_rate * (x - self.grid[i, j])
                else:
                    if((abs(i - line) + abs(j - column)) <= self.radius):
                        self.grid[i, j] = self.grid[i, j] + \
                            self.learning_rate/2 * (x - self.grid[i, j])

    def to_df(self):
        neuron_list = []
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                neuron_list.append(self.grid[i, j])
        return pd.DataFrame(np.array(neuron_list), columns=[
            'slen', 'swid', 'plen', 'pwid'])


def normalize_data(df):
    for column in df.columns:
        if column != 'result':
            df[column] = (df[column] - df[column].min()) / \
                (df[column].max() - df[column].min())
