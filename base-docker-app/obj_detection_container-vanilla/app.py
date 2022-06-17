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

import tensorflow as tf
from tensorflow import keras
from preprocess import preprocess


######## TODO ########
# how does the speech file look like?
# calculate inference time (time it takes for the application to perform speech recognition)


def load_speech_rec_model_from(path: str):
    if not os.path.exists(path):
        print(f"Could not find model under path {path}! Closing application ...")
        exit(-1)
    loaded_model = keras.load_model(path)
    return loaded_model


app = Flask(__name__)
model = load_speech_rec_model_from("saved_model/speech_rec_model")


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
        speech_file = get_file_from_request(accepted_formats)
    except KeyError:
        return Response(status=400)
    except AssertionError:
        return Response(status=400)

    detection_loop(speech_file)

    status_code = Response(status=200)
    return status_code


def get_file_from_request(accepted_types, file_name="file"):
    speech_file = request.files[file_name]
    filetype = speech_file.filename.split('.')[-1]
    if filetype not in accepted_types:
        raise AssertionError
    return speech_file


def detection_loop(speech_file):
    # ?????? how does the speech_file look like ??????
    # if it only contains audio (no label):
    #y_pred = np.argmax(model.predict(speech_file), axis=1)

    # if speech_file is entire dataset and prediction is made on test dataset
    t, test_ds, x, y, z = preprocess(pathlib.Path(speech_file))
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    # calculate the accuracy
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(test_acc)

    # save the prediction
    np.savetxt("prediction.csv", y_pred, delimiter=",")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
