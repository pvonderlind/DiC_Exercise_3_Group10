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
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re
import logging
from datetime import datetime


app = Flask(__name__)

def detection_loop(speech_file):
    pass


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


# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

def get_file_from_request(accepted_types, file_name="file"):
    speech_file = request.files[file_name]
    filetype = speech_file.filename.split('.')[-1]
    if filetype not in accepted_types:
        raise AssertionError
    return speech_file


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
