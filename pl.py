# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Bidirectional

def toColors(dataset):
    b = [2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 29, 28, 31, 33, 35]
    r = [1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]
    
    for c in enumerate(dataset):
        n = c[1]
        i = c[0]
        
        if n in b:
            dataset[i] = 0
        elif n in r:
            dataset[i] = 1
        else:
            dataset[i] = 2
    
    return dataset

def toHomes(dataset):
    for c in enumerate(dataset):
        i = c[0]
        n = c[1]
        
        if n >= 1 and n <= 12:
            dataset[i] = 0
        elif n >= 13 and n <= 24:
            dataset[i] = 1
        elif n >= 25 and n <= 36:
            dataset[i] = 2
        else:
            dataset[i] = 3
    
    return dataset

df = pd.read_csv('data/results.csv')
ns = df['n'].values
ns = toColors(ns)
ns = ns.reshape(-1, 1)

scaler = MinMaxScaler()

ns = scaler.fit_transform(ns)

loopback = 350

x = []
y = []

for v in range(len(ns) - loopback):
    x.append(ns[v:v+loopback,0])
    y.append(ns[v+loopback][0])

x = np.reshape(x, (len(x), 1, loopback))
# y = np.reshape(y, (len(y), 1, 1))
y = np.reshape(y, (len(y), 1))

model = Sequential()
# model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(1, loopback)))
model.add(LSTM(10, input_shape=(1, loopback)))
# model.add(TimeDistributed(Dense(50, activation='relu')))
model.add(Dense(1, activation='relu'))
model.compile(loss='mse', metrics=['accuracy'], optimizer='rmsprop')
model.summary()

model.fit(x, y, epochs=60)

score = model.evaluate(x, y)

print(score)