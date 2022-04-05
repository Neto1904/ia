from distutils.filelist import translate_pattern
from glob import glob
import os
import numpy as np
import pandas as pd
from mlperceptron import MLPerceptron


def load_data():
    file_location = os.path.join('datasets', '*.dat')
    filenames = glob(file_location)
    for f in filenames:
        outfile = open(f, 'r')
        lines = outfile.read()
        inputs = []
        expected = []
        data = lines.split('@data')[1]
        mlp = MLPerceptron([3, 3, 2])
        for line in data.strip().split('\n'):
            line_data = []
            for element in line.strip().split(','):
                if('.' in element):
                    line_data.append(float(element.strip()))
                else:
                    expected.append(translate_pattern(element.strip()))

            inputs.append(line_data)
        inputs_df = pd.DataFrame(np.array(inputs), columns=[
            'slen', 'swid', 'plen', 'pwid'])
        normalize_data(inputs_df)
        mlp.train(inputs, expected)
        outfile.close()


def normalize_data(df):
    for column in df.columns:
        if column != 'result':
            df[column] = (df[column] - df[column].min()) / \
                (df[column].max() - df[column].min())


def translate_pattern(string):
    flowers = {
        'Iris-setosa': [0, 0],
        'Iris-versicolor': [0, 1],
        'Iris-virginica': [1, 0]
    }
    return flowers[string]


load_data()
