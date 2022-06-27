import time
import argparse
import pathlib
import os
import requests
import pandas as pd
from datetime import datetime

parser = argparse.ArgumentParser(description='Train a speech recognition model for a given dataset.')
parser.add_argument('path', type=str, help="The path to the folder containing the training files.")
parser.add_argument('url', type=str, help="URL to the app running the prediction model REST API.")


args = parser.parse_args()
path = args.path
data_dir = pathlib.Path(path)

result_dicts = []
try:
    for path, subdirs, files in os.walk(data_dir):
        for name in files:
            audio_file = pathlib.PurePath(path, name)

            start = time.time()
            params = {"file": audio_file}
            r = requests.get(args.url, params=params)
            end = time.time()
            request_time = end - start

            if r.status_code != 200:
                print(f"An error, a request got back status code {r.status_code}. Exiting application!")
                exit(-1)

            data = r.json()

            file_results = {"request_time": request_time, "file": audio_file,
                            "prediction_time": data['time'], "prediction_label": data['label']}
            result_dicts.append(file_results)
except KeyboardInterrupt:
    print("Interrupted via keyboard. Writing progress to csv and closing ...")
finally:
    timestr = datetime.now().strftime("%Y_%m_%d_%Hh%Mm")
    result_df = pd.DataFrame(result_dicts)
    result_df.to_csv(f"results_{timestr}.csv")
