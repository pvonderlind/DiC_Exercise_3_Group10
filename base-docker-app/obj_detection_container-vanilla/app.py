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
from preprocess import decode_audio, get_spectrogram

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
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
    data_input = request.values.get('input')
    output = request.values.get('output')
    # output = request.form.get('output')

    path = data_input
    filename_image = {}

    accepted_formats = ["wav"]

    if not request.content_type.split(';')[0] == 'multipart/form-data':
        return Response(status=415)

    try:
        speech_file = get_file_binaries_from_request(accepted_formats)
    except KeyError:
        return Response(status=400)
    except AssertionError:
        return Response(status=400)

    predicted_label, prediction_time = detection_loop(speech_file)
    data = {"label": predicted_label, "time": prediction_time}
    status_code = Response(status=200)
    return jsonify(data), status_code


def get_file_binaries_from_request(accepted_types, file_name="file"):
    speech_file = request.files[file_name]
    filetype = speech_file.filename.split('.')[-1]
    if filetype not in accepted_types:
        raise AssertionError
    return speech_file.read()


def detection_loop(audio_binary):
    # TODO: --> Speech_file is a SINGLE .wav speech file in binary format passed via the POST or GET request.
    # if it only contains audio (no label):
    start = time.time()
    audio = convert_binary_audio_to_spectogram(audio_binary)
    y_pred = np.argmax(model.predict(audio), axis=1)
    end = time.time()
    prediction_time = end - start
    return y_pred, prediction_time

    # # if speech_file is entire dataset and prediction is made on test dataset
    # t, test_ds, x, y, z = preprocess(pathlib.Path(speech_file))
    # test_audio = []
    # test_labels = []
    #
    # for audio, label in test_ds:
    #     test_audio.append(audio.numpy())
    #     test_labels.append(label.numpy())
    #
    # test_audio = np.array(test_audio)
    # test_labels = np.array(test_labels)
    #
    # y_pred = np.argmax(model.predict(test_audio), axis=1)
    # y_true = test_labels
    #
    # # calculate the accuracy
    # test_acc = sum(y_pred == y_true) / len(y_true)
    # print(test_acc)
    #
    # # save the prediction
    # np.savetxt("prediction.csv", y_pred, delimiter=",")


# TODO: Fix conversion of audio binary to spectogram with tensor shape (None, 124, 129, 1)
def convert_binary_audio_to_spectogram(audio_binary):
    AUTOTUNE = tf.data.AUTOTUNE

    waveform = decode_audio(audio_binary)
    spectogram_audio = get_spectrogram(waveform)

    audio = np.expand_dims(np.array(spectogram_audio), axis=0)
    return audio


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
