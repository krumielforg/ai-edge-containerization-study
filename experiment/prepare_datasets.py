import os
import idx2numpy
import numpy as np
from onnx import numpy_helper
from PIL import Image
import random
import cv2
import constants

MINST_TESTSET_FILE = "t10k-images-idx3-ubyte"
EMOTION_TESTSET = "./data"
CIFAR_TESTSET = "/Users/krumielf/Downloads/test"
YOLO_TESTSET = "/Users/krumielf/Desktop/Thesis/experiment/onnx-part/val2017"

DEST_MNIST = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/inputs/mnist/pb_dataset_small"
DEST_EMOTION = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/inputs/emotion/pb_dataset"
DEST_CIFAR = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/inputs/cifar10/pb_dataset"
DEST_YOLO = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/inputs/yolov4/pb_dataset"

def prepare_mnist():
    full_path = os.path.join("./", MINST_TESTSET_FILE)
    image_array = idx2numpy.convert_from_file(full_path)
    
    #for j in range(7):
    j = 0
    for i in range(1000):
        
        # preprocess
        image = image_array[i]
        image_data = image.astype(np.float32)
        image_data = np.reshape(image_data, (1, 1, 28, 28))

        # save to pb format
        tensor = numpy_helper.from_array(image_data)
        file_path = os.path.join(DEST_MNIST, f'input_{j}_{i}.pb')
        with open(file_path, 'wb+') as f:
            f.write(tensor.SerializeToString())

def prepare_emotion():
    for dname in os.listdir(EMOTION_TESTSET):
        print(dname)
        dir_path = os.path.join(EMOTION_TESTSET, dname)
        
        subset_images = os.listdir(dir_path)
        random.shuffle(subset_images)
        if dname == 'disgust':
            subset_images = subset_images[:510]
        else:
            subset_images = subset_images[:915]
        for i in range(len(subset_images)):

            # preprocess 
            image_path = os.path.join(dir_path, subset_images[i])
            img = Image.open(image_path)
            img = np.asarray(img.resize((64, 64)))
            img = img / 255.0
            img = img - 0.5
            img = img * 2.0
            img = img.astype('float32')
            image_data = img.reshape((1, 64, 64, 1))
            
            # save to pb format
            tensor = numpy_helper.from_array(image_data)
            file_path = os.path.join(DEST_EMOTION, f'{dname}_input_{i}.pb')
            with open(file_path, 'wb+') as f:
                f.write(tensor.SerializeToString())

def prepare_cifar():
    # https://github.com/scailable/sclbl-tutorials/tree/master/_archive/sclbl-pytorch-onnx

    image_array = os.listdir(CIFAR_TESTSET)
    random.shuffle(image_array)

    for i in range(6000):
        
        # preprocess
        image_path = os.path.join(CIFAR_TESTSET, image_array[i])
        img = Image.open(image_path)
        img = np.array(img)
        img = img.transpose((2, 0, 1))
        img = img/255

        img[0] = (img[0] - 0.4914)/0.2023
        img[1] = (img[1] - 0.4822)/0.1994
        img[2] = (img[2] - 0.4465)/0.2010
    
        image_data = img[np.newaxis,:].astype(np.float32)
        #image_data = img[np.newaxis,:]

        # save to pb format
        image_number = image_array[i].split(".")[0]
        tensor = numpy_helper.from_array(image_data)
        file_path = os.path.join(DEST_CIFAR, f'input_{image_number}.pb')
        with open(file_path, 'wb+') as f:
            f.write(tensor.SerializeToString())

def prepare_yolo():

    image_array = os.listdir(YOLO_TESTSET)
    random.shuffle(image_array)

    for i in range(750):

        img_path = os.path.join(YOLO_TESTSET, image_array[i])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        # save to pb format
        image_number = image_array[i].split(".")[0]
        file_path = os.path.join(DEST_YOLO, f'input_{image_number}.pb')
        tensor = numpy_helper.from_array(img)
        with open(file_path, 'wb+') as f:
            f.write(tensor.SerializeToString())


# to be used on test host
def randomize_input_files():
    print("randomize inputs...")
    for k, v in constants.dataset_files.items():
        file = v[0]   
        if not os.path.exists(file):
            continue

        print(file)
        with open(file, 'r') as f:
            lines = f.readlines()
            random.shuffle(lines)

        with open(file, 'w') as f:
            f.writelines(lines)


#randomize_input_files()
#prepare_mnist()
#prepare_emotion()
#prepare_cifar()
#prepare_yolo()