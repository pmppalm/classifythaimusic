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

import numpy as np
import tensorflow as tf
graph = tf.get_default_graph()

# Import necessary modules
####### end model #######

app = Flask(__name__)

model = load_model('model/SoundContentAnalysis_model.h5')


def prediction_sound(file_name):

    def genre(i):
        if i == 0:
            return 'ฮิปฮอป'
        elif i == 1:
            return 'ป๊อป'
        elif i == 2:
            return 'คลาสสิก'
        elif i == 3:
            return 'ร็อค'

    sound = 'เล่นของสูง'
    pathfile = '../static/test_model/'+file_name
    D_test = []  # test dataset
    y, sr = librosa.load('static/test_model/'+file_name,
                         offset=120.0, duration=10)
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