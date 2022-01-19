from timeit import default_timer as timer
import sys
import time

import constants as cnst
import model as m


def init_onnx_model(model_name):
    #print(model_name)
    if model_name == 'mnist':
        return m.MNIST()
    elif model_name == 'cifar10':
        return m.CIFAR10()
    elif model_name == 'emotion':
        return m.Emotion()
    elif model_name == 'yolov4':
        return m.YOLOv4(m.COCO_DATASET_CLASESS)
    else:
        return None


def run_onnx_image(model, pb_path):
    image_data = model.read_pb(pb_path)
    return model.predict(image_data)


def get_onnx_input(model: m.BaseModel):
    return model.get_inputs()


def run_onnx(depl_type, model_name):

    # load model
    model = init_onnx_model(model_name)

    if model == None:
        print ("failed to init model")
        return

    # load input list - a list of paths containing .pb files
    inputs = get_onnx_input(model)
    print("inputs:", len(inputs))
    #print(inputs)


    # inftime_log = open(cnst.get_log_name(cnst.PerfMonitor.INFTIME, depl_type, model_name), "x")
    # for j in range(cnst.NUM_IMAGES):
    for j in range(len(inputs)):
        # i_time = 0
        _, i_time = run_onnx_image(model, inputs[j])
        # inftime_log.write(cnst.get_log_line(cnst.PerfMonitor.INFTIME, i, i_time, str(j)))
        # inftime_log.flush()

    # exectime_log.close()


def run_docker(name, depl, model, row_id, run_id, lpath):
    
    import docker
    client = docker.from_env()
    client.containers.run(image=cnst.DOCKER_IMAGE,
                          command=['python3', 'measure.py',
                                   f'{depl}', f'{model}', f'{row_id}', f'{run_id}', f'{lpath}'],
                          detach=False,
                          name=name,
                          volumes=cnst.MOUNT_VOLUMES,
                          working_dir=cnst.MOUNT_WORKING_DIR,
                          remove=True #Remove the container when it has finished running
                          )

