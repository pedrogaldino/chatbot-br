# -*- coding: utf-8 -*-

import json
import nltk
import numpy as np
from gensim import models
import pickle

# w2v_en = models.Word2Vec.load('word2vec.bin');
# w2v_br = models.KeyedVectors.load_word2vec_format("wiki.pt.trigram.vector", binary=True)

w2v = models.KeyedVectors.load_word2vec_format("wiki.pt.trigram.vector", binary=True)

file = open('conversation_br.json');
data = json.load(file)
cor = data["conversations"];

x=[]
y=[]

for i in range(len(cor)):
    for j in range(len(cor[i])):
        if j<len(cor[i])-1:
            x.append(cor[i][j]);
            y.append(cor[i][j+1]);

tok_x=[]
tok_y=[]

for i in range(len(x)):
    tok_x.append(nltk.word_tokenize(x[i].lower()))
    tok_y.append(nltk.word_tokenize(y[i].lower()))
    
sentend = np.ones((400,),dtype=np.float32)

vec_x=[]
for sent in tok_x:
    sentvec = [w2v[w] for w in sent if w in w2v.vocab]
    vec_x.append(sentvec)
    
    
vec_y=[]
for sent in tok_y:
    sentvec = [w2v[w] for w in sent if w in w2v.vocab]
    vec_y.append(sentvec)
    

for tok_sent in vec_x:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_x:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)    
            
for tok_sent in vec_y:
    tok_sent[14:]=[]
    tok_sent.append(sentend)
    

for tok_sent in vec_y:
    if len(tok_sent)<15:
        for i in range(15-len(tok_sent)):
            tok_sent.append(sentend)
            
pickle.dump([vec_x,vec_y], open('conversation_br.pickle','wb'))