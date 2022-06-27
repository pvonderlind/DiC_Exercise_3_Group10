# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tflite_runtime.interpreter as tflite
import platform
import datetime
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
import logging
from datetime import datetime
import pathlib
import argparse
from preprocess import decode_audio, get_spectrogram, preprocess

import tensorflow as tf
from tensorflow import keras


######## TODO ########
# how does the speech file look like?
# calculate inference time (time it takes for the application to perform speech recognition)


def load_speech_rec_model_from(path: str):
    if not os.path.exists(path):
        print(f"Could not find model under path {path}! Closing application ...")
        exit(-1)
    loaded_model = keras.models.load_model(path)
    print(f"Successfully loaded pretrained tensorflow model from path {path}")
    return loaded_model


parser = argparse.ArgumentParser(description="Run a Flask API offering a REST API for speech recognition of a"
                                             "pretrained deep learning model in tensorflow.")
parser.add_argument("model_path", type=str, help="Path to the pretrained tensorflow model in .pb format.")
args = parser.parse_args()
model_path = args.model_path

app = Flask(__name__)
model = load_speech_rec_model_from(model_path)


# routing http posts to this method
@app.route('/api/detect', methods=['GET'])
def main():
    file_path = request.args['file']

    accepted_formats = ["wav"]

    try:
        speech_file = get_file_tensor_from_request(file_path, accepted_formats)
    except KeyError:
        return Response(status=400)
    except AssertionError:
        return Response(status=400)

    predicted_label, prediction_time = detection_loop(speech_file)
    data = {"label": predicted_label, "time": prediction_time}
    return data


def get_file_tensor_from_request(path, accepted_types):
    filetype = path.split('.')[-1]
    if filetype not in accepted_types:
        raise AssertionError
    return open_local_file(path)


def open_local_file(path):
    try:
        return tf.io.read_file(path)
    except Exception:
        raise AssertionError


def detection_loop(audio_binary):
    # TODO: --> Speech_file is a SINGLE .wav speech file in binary format passed via the POST or GET request.
    # if it only contains audio (no label):
    start = time.time()
    audio = convert_binary_audio_to_spectogram(audio_binary)
    y_pred = np.argmax(model.predict(audio), axis=1)

    # load commands to get the actual word of the prediction
    commands = commands['backward', 'follow', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']
    prediction = commands[y_pred[0]]

    end = time.time()
    prediction_time = end - start
    return prediction, prediction_time


def convert_binary_audio_to_spectogram(audio_binary):
    waveform = decode_audio(audio_binary)
    spectogram_audio = get_spectrogram(waveform)

    audio = np.expand_dims(np.array(spectogram_audio), axis=0)
    return audio


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
