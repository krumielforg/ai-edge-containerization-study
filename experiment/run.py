import sys
import time
from timeit import default_timer as timer

import constants as cnst
import helpers

# this is in a separate file so that we can get the pid of the process to measure
def run(depl_type, model_name):
    print(depl_type, model_name)
    if depl_type == cnst.Deployment.ONNX.name or depl_type == cnst.Deployment.DOCKER.name:
        helpers.run_onnx(depl_type, model_name)
    else:
        print('not supported')

def run_dummy_for_testing_measurements(path):
    for i in range(10):
        time.sleep(1)


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2])
    # run_dummy_for_testing_measurements(sys.argv[1])
