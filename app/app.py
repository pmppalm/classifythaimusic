from flask import Flask, request, jsonify, render_template, url_for, redirect
import urllib.parse
# from flask_cors import CORS
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
import poem
from flask_cors import CORS

import numpy as np
import tensorflow as tf
from poem import prediction_poem
import integation
from integation import prediction
from pred import prediction_sound
graph = tf.get_default_graph()


app = Flask(__name__)
CORS(app)

# CORS(app)

model = load_model('SoundContentAnalysis13.h5')


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.errorhandler(404)
def not_found(e):
    return render_template('error_page.html'), 404


@app.route('/pred_lyrics', methods=['GET'])
def pred_lyrics():
    # Main page
    return render_template('index_lyrics.html')


@app.route('/all', methods=['GET'])
def all():
    # Main page
    return render_template('index2.html')


@app.route('/pred_sound', methods=['GET'])
def pred_sound():
    # Main page
    return render_template('index_sound.html')


@app.route('/api_sound', methods=['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def api_sound():
    if request.method == "GET":
        args = request.query_string.decode('UTF-8')
        return jsonify({
            "prediction_sound": pred.prediction_sound(urllib.parse.unquote(args)+'.mp3')
        })


@app.route('/api_lyrics', methods=['GET', 'POST', 'PATCH', 'PUT', 'DELETE'])
def api_lyrics():
    if request.method == "GET":
        args = request.query_string.decode('UTF-8')
        return jsonify({
            "prediction_lyrics": prediction_poem(urllib.parse.unquote(args))
        })


@app.route('/success', methods=['POST', 'GET'])
def success():
    if request.method == 'POST':
        message = request.form['message']
        inputs = message+'.mp3'
        p = integation.prediction(inputs)
        l = prediction_poem(message)
    return render_template("success.html", name=l['name'], prediction=p['pred_class'], pathfile=p['pathfile'], hiphop=p['HipHop'], pop=p['Pop'], rock=p['Rock'], classic=p['Classic'], sad=l['sad'], love=l['love'], rave=l['rave'], arousing=l['arousing'], strong=l['strong'], artist=l['artist'],lyrics=l['lyrics'])

@app.route('/lyrics_show', methods=['POST', 'GET'])
def lyrics_show():
    if request.method == 'POST':
        message = request.form['message']
        inputs = message+'.mp3'
        p = integation.prediction(inputs)
        l = prediction_poem(message+'.txt')
    return render_template("test.html", name=l['name'], prediction=p['pred_class'], pathfile=p['pathfile'], hiphop=p['HipHop'], pop=p['Pop'], rock=p['Rock'], classic=p['Classic'], sad=l['sad'], love=l['love'], rave=l['rave'], arousing=l['arousing'], strong=l['strong'], artist=l['artist'],lyrics=l['lyrics'])


@app.route('/poem', methods=['POST'])
def poem():
    if request.method == 'POST':
        # message = request.form['message']
        f = request.files['file']
        f.save('static/test_model/'+f.filename)
        p = integation.prediction_poem(f.filename)
    return render_template("show2.html", name=p['sentence'], prediction=p['pred_class'], sad=p['sad'], love=p['love'], rave=p['rave'], arousing=p['arousing'], strong=p['strong'])


@app.route('/sound', methods=['POST'])
def sound():
    if request.method == 'POST':
        f = request.files['file']
        f.save('static/test_model/'+f.filename)
        p = prediction_sound(f.filename)
    return render_template("show1.html", name=p['file_name'], prediction=p['pred_class'], pathfile=p['pathfile'], hiphop=p['HipHop'], pop=p['Pop'], rock=p['Rock'], classic=p['Classic'])


if __name__ == '__main__':
    # run server with ip server
    app.run(host="localhost", port=3000, threaded=False)
