# DIC Exercise 3
## Docker Usage
To build the container itself, please make use of the bash script ``` build_docker.sh```.
The script has to be executed exactly in the folder where it is contained.

In order to launch the containers, please use the bash scripts
```run_docker.sh``` and ```run_docker_cluster.sh``` for local and
on cluster execution respectively. The only difference between the executables
is the port assignment (5000 for local, 5010 for the cluster).

## Running the app

Running the app is easy, as the containers mount the directory where the bash scripts are located
as the /app folder of the container. Upon using on of the run_docker bash files, the Flask app
will automatically started, using the pre-trained speech recognition model in saved_models/speech_rec_model.

### Datasets
The datasets have to manually place inside of this directory (ergo the one where the app.py and docker run files are
 contained). Please do this **before** running the build and run bash scripts in any case!!

As per TUWEL forum, a tutor stated the way to go for file upload time measurement is to just record the required upload
time. So please, take note of this when uploading the whole dataset directories to the cluster!!

## Running the time measurement

Inside this folder a script ```measure_time.py``` is contained. It takes a local path to a dataset (z.B. BIG,SMALL,MEDIUM) rerelative to the scripts location. Note that this **has to be exactly the same path that is usable by the container** -->
This mainly means you have to execute the measure_time script from the same folder as the run_docker scripts. 
THe second parameter of the script is the URL to the docker container. Since we compare local and remote execution, please
use the following two URLs:
* http://localhost:5000/api/detect
* http://s25.lbd.hpc.tuwien.ac.at:5010/api/detect

A call to measure timing for the BIG dataset could be as follows to test the remote execution:

```python measure_time.py BIG http://s25.lbd.hpc.tuwien.ac.at:5010/api/detect```
