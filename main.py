import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import keras
from fuzzy_layer import FuzzyLayer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import plot_model
import random as rnd
from matplotlib.patches import Ellipse
from sklearn import datasets
import tensorflow as tf

removed_cols = ['status', 'name', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']

def main():
    park = pd.read_csv("data/parkinsons.data")
    target = park['status'].to_numpy()
    raw_data = park.loc[:, ~park.columns.isin(removed_cols)].to_numpy()
    data = raw_data
    Y = []
    for y in target:
        tmp = np.zeros(2)
        tmp[y] = 1
        Y.append(tmp)

    x_train, x_test, y_train, y_test = train_test_split(data, Y, test_size=0.1)
    K = 25
    indices = rnd.sample(range(len(x_train)), K)

    f_layer = FuzzyLayer(K, initial_centers=np.transpose(np.array([x_train[i] for i in indices])),
                         input_dim=data.shape[1])

    model = Sequential()
    model.add(f_layer)
    model.add(Dense(15, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='mean_squared_error',
                  optimizer='rmsprop',
                  metrics=['binary_accuracy'])

    model.fit(np.array(x_train),
              np.array(y_train),
              epochs=5000,
              verbose=1,
              batch_size=10)

    score = model.evaluate(np.array(x_test), np.array(y_test), verbose=True)
    print(score)
    model.save('model')
    weights = f_layer.get_weights()
    # print(weights)

    #predictions
    a = park.loc[:, ~park.columns.isin(removed_cols)]
    b = a.to_numpy()[0]
    print(model.predict(np.expand_dims(b, axis=0)))


    #graphic output
    plt.ion()
    plt.show()
    plt.clf()
    plt.title('Parkinson\'s Disease')
    plt.ylabel('x[0]')
    plt.xlabel('x[1]')
    plt.scatter([a[0] for a in x_train], [a[1] for a in x_train], c=(0, 0, 0), alpha=0.5, s=1)
    for i in range(0, K):
        ellipse = Ellipse((weights[0][0][i], weights[0][1][i]), weights[1][0][i], weights[1][1][i], color='r',
                          fill=False)
        ax = plt.gca()
        ax.add_patch(ellipse)

    plt.scatter(weights[0][0], weights[0][1], c=(1, 0, 0), alpha=0.8, s=15)
    plt.show()
    plt.pause(1200)

if __name__ == '__main__':
    main()