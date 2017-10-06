# -*- coding: utf-8 -*-

from scipy import spatial
import numpy as np
import gensim
import nltk
from keras.models import load_model

model = load_model('LSTM500.h5')
# mod = gensim.models.Word2Vec.load('word2vec.bin');

w2v = gensim.models.KeyedVectors.load_word2vec_format("wiki.pt.trigram.vector", binary=True)

while(True):
    x = input("Enter the message: ");
    
    sentend=np.ones((400,), dtype=np.float32) 

    sent=nltk.word_tokenize(x.lower())
    sentvec = [w2v[w] for w in sent if w in w2v.vocab]

    sentvec[14:] = []
    sentvec.append(sentend)
    
    if len(sentvec) < 15:
        for i in range(15-len(sentvec)):
            sentvec.append(sentend) 

    sentvec = np.array([sentvec])

    predictions = model.predict(sentvec)
    outputlist = [w2v.most_similar([predictions[0][i]])[0][0] for i in range(15)]
    output = ' '.join(outputlist)
    
    print(output)