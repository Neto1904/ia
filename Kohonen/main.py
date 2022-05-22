from glob import glob
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import seaborn as sns
from kohonen import Kohonen
import matplotlib.pyplot as charts


def load_data(width=10, length=10):
    file_location = os.path.join('datasets', '*.dat')
    filenames = glob(file_location)
    kohonen = Kohonen(width, length)
    for f in filenames:
        outfile = open(f, 'r')
        lines = outfile.read()
        inputs = []
        expected = []
        data = lines.split('@data')[1]
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
        input_list = inputs_df.values.tolist()
        kohonen.fit(input_list)
        som_neurons = kohonen.to_df()
        apply_kmeans(som_neurons)
        outfile.close()


def translate_pattern(string):
    flowers = {
        'Iris-setosa': [0, 0],
        'Iris-versicolor': [0, 1],
        'Iris-virginica': [1, 0]
    }
    return flowers[string]


def normalize_data(df):
    for column in df.columns:
        if column != 'result':
            df[column] = (df[column] - df[column].min()) / \
                (df[column].max() - df[column].min())


def apply_kmeans(dataframe):
    kmeans = KMeans(n_clusters=3)
    df_array = dataframe.values.tolist()
    label = kmeans.fit_predict(dataframe)
    print('Classification:', label)
    sns.scatterplot(data=dataframe, x="slen", y="swid")
    charts.scatter(kmeans.cluster_centers_[:, 0],
                   kmeans.cluster_centers_[:, 1], s=100, c='red')
    charts.show('figures/kmeans')


def main():
    load_data(3, 3)
    # load_data(5, 5)
    # load_data(10, 10)


main()
