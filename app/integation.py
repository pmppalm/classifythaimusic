from pythainlp import word_tokenize
from keras import backend as K
from keras.preprocessing import sequence
from keras.preprocessing.sequence import skipgrams
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.layers.merge import Dot
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import MaxPooling1D, Conv1D, SeparableConv1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Embedding, Reshape, Activation, Input, Dense, Flatten, Dropout, LSTM
from keras.models import Sequential, Model
import poem
import sys
import requests
import collections
import re
import glob
import math
import csv
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from flask import Flask, redirect, url_for, request, render_template, jsonify
import os
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
######## model #############
import keras
from keras import layers
from keras.layers import Activation, Dense, Dropout, Conv1D, Conv2D, Flatten, BatchNormalization, ZeroPadding2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling1D, AveragePooling2D, Input, Add
from keras.models import Sequential
from keras import regularizers
from keras.optimizers import SGD
import keras.backend as K
from keras.models import load_model
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display
from numpy import argmax
import pandas as pd
import random
import pred

import integation

import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()

# Import necessary modules
####### end model #######


#########################
app = Flask(__name__)

model = load_model('model/SoundContentAnalysis_model.h5')
model2 = load_model('model/LyricsAnalysis_model.h5')


def prediction(file_name):

    def genre(i):
        if i == 0:
            return 'Hip-Hop'
        elif i == 1:
            return 'Pop'
        elif i == 2:
            return 'Classical'
        elif i == 3:
            return 'Rock'

    sound = 'เล่นของสูง'
    pathfile = '../static/sound/'+file_name
    D_test = []  # test dataset
    y, sr = librosa.load('static/sound/'+file_name, offset=120.0, duration=10)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

    X_test = np.array([ps.reshape((128, 431, 1))])
    with graph.as_default():
        yhat = model.predict_classes(X_test)
        yped = model.predict(X_test)
        result = str(genre(yhat))
        h = yped[0][0]*100
        p = yped[0][1]*100
        c = yped[0][2]*100
        r = yped[0][3]*100

        results = {'file_name': file_name,
                   "pred_class": result,
                   "HipHop": float("%.2f" % (h)),
                   "Pop": float("%.2f" % (p)),
                   "Classic": float("%.2f" % (c)),
                   "Rock": float("%.2f" % (r)),
                   "pathfile": pathfile
                   }
        print(results)
    return results


def prediction_poem(data):
    file_text = data
    filename = 'static/test_model/'+data
    with open(filename, 'r', encoding="utf-8") as f:
        input_text = f.read().split('\n')

    countLyrics = len(input_text)
    artist = input_text[0]
    data1 = []
    countWord = []
    count = 0

    for t in input_text:
        count += 1
        data = word_tokenize(t)
        data1.append(data)
        countWord.append(len(data))

    data_t = []
    data_max = []
    for t in data1:
        s = 29-len(t)
        s2 = 0
        if len(t) <= 29:
            for s2 in range(s):
                t += " "
            data_t.append(t)
            data_max.append(len(t))

    filename = 'model/poemlyrics.txt'
    with open(filename, 'r', encoding="utf-8") as f:
        input_text = f.read().split()

    def create_index(input_text):

        words = [word for word in input_text]
        word_count = list()
        word_count.extend(collections.Counter(
            words).most_common(len(set(words))))
        word_count.append(("UNK", 0))

        dictionary = dict()
        for word in word_count:
            dictionary[word[0]] = len(dictionary)

        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        data = list()
        for word in input_text:
            data.append(dictionary[word])

        return data, dictionary, reverse_dictionary

    dataset, dictionary, reverse_dictionary = create_index(input_text)
    del input_text
    vocab_size = len(dictionary)

    dim_embedddings = 100

    w_inputs = Input(shape=(1, ), dtype='int32')
    w = Embedding(vocab_size, dim_embedddings)(w_inputs)

    c_inputs = Input(shape=(1, ), dtype='int32')
    c = Embedding(vocab_size, dim_embedddings)(c_inputs)

    o = Dot(axes=2)([w, c])
    o = Reshape((1,), input_shape=(1, 1))(o)

    o = Activation('sigmoid')(o)

    SkipGram = Model(inputs=[w_inputs, c_inputs], outputs=o)
    SkipGram.compile(loss='binary_crossentropy', optimizer='adam')

    SkipGram.load_weights('model/skipgram100_weights.h5')
    final_embeddings = SkipGram.get_weights()[0]

    wordCount = 29

    def lookupWord2Vec(data):
        inputs = []
        for poemCount in range(len(data)):
            poem = []
            for count in range(wordCount):
                search = data[poemCount][count]
                if search != '' and search in dictionary.keys():
                    poem = poem + [dictionary[search]]
                else:
                    poem = poem + [dictionary['UNK']]
            inputs.append(poem)
        return inputs

    train_vectors = lookupWord2Vec(data_t)
    train_vectors = np.asarray(train_vectors)

    def genre(i):
        if i == 1:
            return 'อกหัก'
        elif i == 2:
            return 'รัก'
        elif i == 3:
            return 'เพ้อเจ้อ'
        elif i == 4:
            return 'ปลุกใจ'
        elif i == 5:
            return 'รุนแรง'

    yped = model2.predict(train_vectors)
    yhat = model2.predict_classes(train_vectors)

    sad = 0
    love = 0
    rave = 0
    arousing = 0
    strong = 0
    for s in yhat:
        if s == 0:
            sad += 1
        elif s == 1:
            love += 1
        elif s == 2:
            rave += 1
        elif s == 3:
            arousing += 1
        elif s == 4:
            strong += 1

    class_pred = [sad, love, rave, arousing, strong]
    class_pred2 = max(class_pred)
    if class_pred2 == class_pred[0]:
        result = "อกหัก"
    elif class_pred2 == class_pred[1]:
        result = "รัก"
    elif class_pred2 == class_pred[2]:
        result = "เพ้อเจ้อ"
    elif class_pred2 == class_pred[3]:
        result = "ปลุกใจ"
    elif class_pred2 == class_pred[4]:
        result = "รุนแรง"

    results = {'sentence': file_text,
               "count_lyrics": countLyrics,
               "pred_class": result,
               "sad_count": sad,
               "love_count": love,
               "rave_count": rave,
               "arousing_count": arousing,
               "strong_count": strong,
               "sad": "%.2f" % ((sad/countLyrics)*100),
               "love": "%.2f" % ((love/countLyrics)*100),
               "rave": "%.2f" % ((rave/countLyrics)*100),
               "arousing": "%.2f" % ((arousing/countLyrics)*100),
               "strong": "%.2f" % ((strong/countLyrics)*100)
               }

    print(results)

    return results
