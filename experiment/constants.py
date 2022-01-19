import os
from datetime import datetime
from enum import Enum
from model import INPUTS_DIR


class Deployment(Enum):
    ONNX = 1
    SCLB = 2
    DOCKER = 3


class PerfMonitor(Enum):
    CPUMEM = 1
    INFTIME = 2
    EXECTIME = 3
    HVPM = 4


SCLB_BINARY = "./sclbl-bin"
MODELS_PATH = "./models"

NUM_RUNS = 10
NUM_IMAGES = 10

LOG_FOLDER = "./perflogs"
# we ignore inference time for now

LOG_FILE = "_log.txt"
DATETIME_FMT = "%m%d%Y_%H:%M:%S:%f"

DOCKER_IMAGE = "rhampau/onnxenv:4.0"
MOUNT_WORKING_DIR = "/home/ubuntu/rpi4/experiment"
#MOUNT_WORKING_DIR = "/home/experiment"
#LOCAL_WORKIMG_DIR_MAC = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/"
LOCAL_WORKIMG_DIR = "/home/ubuntu/rpi4/experiment"
MOUNT_VOLUMES = {LOCAL_WORKIMG_DIR: {'bind': MOUNT_WORKING_DIR, 'mode': 'rw'}}

# model: (dataset_input_list_file, num_of_inputs)
dataset_files_rpi4 = {
    "mnist":(os.path.join(INPUTS_DIR, "mnist_dataset.txt"), 70000),
    "emotion":(os.path.join(INPUTS_DIR, "emotion_dataset.txt"), 4000),
    "cifar10":(os.path.join(INPUTS_DIR, "cifar10_dataset.txt"), 4000),
    "yolov4":(os.path.join(INPUTS_DIR, "yolov4_dataset.txt"), 700)
}

dataset_files = {
    "mnist":(os.path.join(INPUTS_DIR, "mnist_dataset.txt"), 30000),
    "emotion":(os.path.join(INPUTS_DIR, "emotion_dataset.txt"), 1800),
    "cifar10":(os.path.join(INPUTS_DIR, "cifar10_dataset.txt"), 2000),
    "yolov4":(os.path.join(INPUTS_DIR, "yolov4_dataset.txt"), 300)
}


def get_input_path(model):
    return dataset_files[model][0]


def get_wasm_path(model):
    return os.path.join(MODELS_PATH, f'{model}', 'model.wasm')

def get_aot_path(model):
    return os.path.join(MODELS_PATH, f'{model}', f'{model}-aarch64.aot')


def get_now():
    return str(datetime.now().strftime(DATETIME_FMT))


def get_log_line(mtype, runid, time, start=0, end=0, image_name=""):
    if mtype == PerfMonitor.INFTIME:
        return ""
    elif mtype == PerfMonitor.EXECTIME:
        return get_now() + f' run:{runid} duration (s):{time} start:{start} end:{end}\n'

def get_log_name(lpath, mtype, depl, model, row_id, run_id):
    if mtype == PerfMonitor.INFTIME:
        return ""
    elif mtype == PerfMonitor.EXECTIME:
        return os.path.join(lpath, f'{depl}_{model}' + LOG_FILE)
    elif mtype == PerfMonitor.CPUMEM:
        return os.path.join(lpath, f'{run_id}_{depl}_{model}_' + get_now() + LOG_FILE)
    elif mtype == PerfMonitor.HVPM:
        return os.path.join(lpath, f'{depl}_{model}_' + get_now() + LOG_FILE)
    else:
        return ""

def get_log_folder_name():
    return os.path.join(MOUNT_WORKING_DIR, LOG_FOLDER, get_now() + "_logs")
