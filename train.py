# -*- coding: utf-8 -*-

import pickle
import numpy as np
from keras.models import Sequential
from gensim import models
from keras.layers import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split

with open('conversation_br.pickle', 'rb') as f:
    vec_x, vec_y = pickle.load(f)

vec_x = np.array(vec_x, dtype=np.float32)
vec_y = np.array(vec_y, dtype=np.float32)

x_train, x_test, y_train, y_test = train_test_split(vec_x, vec_y, test_size=0.2, random_state=1)

model=Sequential()
model.add(LSTM(400, input_shape=x_train.shape[1:], return_sequences=True, kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal', activation='sigmoid'))
model.add(LSTM(400, input_shape=x_train.shape[1:], return_sequences=True, kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal', activation='sigmoid'))
model.add(LSTM(400, input_shape=x_train.shape[1:], return_sequences=True, kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal', activation='sigmoid'))
model.add(LSTM(400, input_shape=x_train.shape[1:], return_sequences=True, kernel_initializer='glorot_normal', recurrent_initializer='glorot_normal', activation='sigmoid'))
model.compile(loss='cosine_proximity', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_500.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_1000.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_1500.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_2000.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_2500.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_3000.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_3500.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_4000.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_4500.h5');

model.fit(x_train, y_train, epochs=500, validation_data=(x_test, y_test))
model.save('LSTMBR_5000.h5');

predictions = model.predict(x_test)

# mod = gensim.models.Word2Vec.load('word2vec.bin');   

w2v = models.KeyedVectors.load_word2vec_format("wiki.pt.trigram.vector", binary=True)

[w2v.most_similar([predictions[10][i]])[0] for i in range(15)]