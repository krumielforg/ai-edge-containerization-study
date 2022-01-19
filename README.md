# Replication Package
_An empirical study on the performance and energy consumption of AI containerization strategies on the Edge_

The full dataset including raw data, data analysis python scripts and automatization scripts produced during the study are available below.

This study has been designed, developed, and reported by the following investigators:
- Raluca Maria Hampău (Vrije Universiteit Amsterdam)
- Ivano Malavolta (Vrije Universiteit Amsterdam)
- ...

#### Overview of the replication package scripts

```
- replication_package
  - datanalysis                           Data analysis scripts (used on host) - simply use: python3 <name_of_script>
   ├── cpumem.py                          Parses raw data for execution, cpu and memory measures (should be used before performing statistical tests)
   ├── energy.py                          Parses raw data for energy (should be used before performing statistical tests)
   ├── statistical_analysis.py            Performs statistical tests per containerization strategy
   └── statistical_analysis_subjects.py   Performs statistical tests per subject 
  
  - experiment                            Experiment automation scripts (used on both host & test device)
   ├── constants.py                       Helper script
   ├── extract_imagenet.py                Helper script to preprocess inputs
   ├── helpers.py                         Helper script to run onnxruntime
   ├── hvpm.py                            Helper script to sample power measures
   ├── main.py                            Main script <- starting point of the execution
   ├── measure.py                         Starts profiling and runs the containerization strategy
   ├── model.py                           Loads models into ONNX Runtime and was also used to experiment with the models separately
   ├── pipsize.py                         Helper script to check python package size
   ├── prepare_datasets.py                Helper script to prepare datasets - converts images to .pb
   ├── run.py                             Helper script to run onnxruntime
   ├── run_docker.py                      Helper script to run docker
   
   ├── rows.csv                           All combinations of subject - treatment performed
   ├── measure.sh                         Bash script to measure cpu and memory on test device
   
  - raw_logs                              Raw data collected
```

#### Overview on running the experiment

![alt text](https://github.com/krumielf/vuthesis/blob/main/replication_package/files/Screenshot%202022-01-18%20at%2021.02.51.png)

Pre-requisites:

- [Raspberry Pi] Install the necessary packages (check Dockerfile for a detailed list on those)
- [Raspberry Pi & local device (MPB)] Copy experiment folder scripts on both machines
    - The location of the scripts on the device should be at the path specified by MOUNT_WORKING_DIR from experiment/constants.py
    - Additional details about the paths & experiment constants can be found/edited in experiment/constants.py
    - [Raspberry Pi] The models and inputs folder should be inside the experiment folder on the target device
- [Raspberry Pi] Build (using the Dockerfile) or download the docker image beforehand ([https://hub.docker.com/repository/docker/rhampau/onnxenv](https://hub.docker.com/repository/docker/rhampau/onnxenv))
- [Raspberry Pi & local device ] Generate a passphrase-less SSH key and push it to the test device ([https://serverfault.com/questions/241588/how-to-automate-ssh-login-with-password](https://serverfault.com/questions/241588/how-to-automate-ssh-login-with-password))

Running the experiment:

- On local machine run: python3 main.py
- One can customise the number\type of treatments by editing the rows.csv file

Post-requisites:

- For new raw data, manually copy all logs to raw_logs folder, following the directory tree format from the replication package
