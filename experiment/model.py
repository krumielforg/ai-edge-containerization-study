import os
import sys
import json
import idx2numpy
import numpy as np
import onnx
import onnxruntime
import cv2
import random
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches


from timeit import default_timer as timer
from datetime import timedelta
from PIL import Image
from enum import Enum
from onnx import numpy_helper
import constants


# nltk.download('punkt')
onnxruntime.set_default_logger_severity(3)


MODELS_DIR = "./models"
INPUTS_DIR = "./inputs"

MINST_TESTSET_FILE = "t10k-images-idx3-ubyte"
COCO_DATASET_CLASESS = "coco_classes.txt"
IMAGENET_DATASET_CLASSES = "imagenet_classes.txt"


class ModelFamily(Enum):
    COMPUTER_VISION = 1
    NATURAL_LANGUAGE_PROCESSING = 2


class BaseModel(object):

    def __init__(self, directory, name, model_type):
        self.name = name
        self.directory = directory
        self.type = model_type

        if directory != 'gpt2':
            model = onnx.load(os.path.join(
                MODELS_DIR, self.directory, self.name))
            self.session = onnxruntime.InferenceSession(
                model.SerializeToString())
            #print(self.session.get_device())

    def get_name(self):
        return self.name

    def preprocess(self):
        return 'not implemented'

    def predict(self):
        return 'not implemented'

    def post_process(self):
        return 'not implemented'
    
    def get_inputs(self):
        #return [os.path.join(INPUTS_DIR, self.directory, 'pb_dataset', file) for file in os.listdir(os.path.join(INPUTS_DIR, self.directory, 'pb_dataset'))]
        input_file = constants.dataset_files[self.directory][0]
        with open(input_file, 'r') as f:
            #lines = f.readlines()
            lines = f.read().splitlines() 
             
        return lines[:constants.dataset_files[self.directory][1]]

    def read_pb(self, pb_path):
        tensor = onnx.TensorProto()
        with open(pb_path, 'rb') as f:
            tensor.ParseFromString(f.read())
        return numpy_helper.to_array(tensor)


# WARNING! Input image should be a white number on a black background.
class MNIST(BaseModel):
    def __init__(self):
        BaseModel.__init__(self, 'mnist', 'model.onnx',
        #BaseModel.__init__(self, 'mnist_big', 'mnist.onnx',
                           ModelFamily.COMPUTER_VISION)

    def prepare_dataset(self, path=""):
        full_path = os.path.join(path, MINST_TESTSET_FILE)
        image_array = idx2numpy.convert_from_file(full_path)
        return image_array

        # plt.imshow(image_array[5])  
        # plt.show()
        # print(len(image_array))
        # img = image_array[5].astype(np.float32)
        # img = np.reshape(img, (1, 1, 28, 28))
        # return img   

    def preprocess(self, image_path="", image=None, from_dataset=True):

        dest_path = os.path.join("/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/models/mnist", "mnist_test.pb")    

        # we exepect it from the dataset 
        if from_dataset:
            image_data = image.astype(np.float32)  
        # we expect and image_path as input
        else:
            image_data = cv2.imread(image_path)
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            image_data = cv2.resize(image_data, (28, 28)).astype(np.float32)/255
        
        #this is for mnist-8.onnx
        image_data = np.reshape(image_data, (1, 1, 28, 28))
 
        #this is for the mnist from sclbl repo
        #image_data = np.reshape(image_data, (1, 28, 28, 1))

        # Convert the Numpy array to a TensorProto
        tensor = numpy_helper.from_array(image_data)
        # print('TensorProto:\n{}'.format(tensor))

        # Save the TensorProto
        with open(dest_path, 'wb+') as f:
            f.write(tensor.SerializeToString())

        return image_data

    def predict(self, image_data, post_process=False):

        inputs = {self.session.get_inputs()[0].name: image_data}

        start = timer()
        preds = self.session.run(None, inputs)
        inference_time = timer() - start

        if post_process:
            print("Preds:", preds)
            print("Predicted digit:", np.argmax(preds[0]))
            print("Inference duration (s):", inference_time)

        return preds[0], inference_time

class Resnet(BaseModel):
    def __init__(self, classes_file):
        BaseModel.__init__(self, 'resnet', 'resnet18-v1-7.onnx',
                           ModelFamily.COMPUTER_VISION)

        self.classes = [l.rstrip() for l in open(classes_file, 'r')]

    def prepare_dataset(self, path):
        count = 0 
        dest = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/picked_imagenet"

        if os.path.exists(dest):
            paths = [os.path.join(dest, f) for f in os.listdir(dest)]
            return paths

        os.mkdir(dest)
        class_ids = list(map(lambda x: x.split()[0], self.classes))
        for subset in os.listdir(path):
            if subset not in class_ids:
                continue

            subset_images = os.listdir(os.path.join(path, subset))
            random.shuffle(subset_images)
            picked_images = subset_images[:15]
            for img in picked_images:
                full_path = os.path.join(path, subset, img)
                shutil.copyfile(full_path, os.path.join(dest, img))

        return os.listdir(dest)
            

    def preprocess(self, image_path):
        img = Image.open(image_path)
        img = np.array(img.convert("RGB"))
        img = img / 255.
        img = cv2.resize(img, (256, 256))
        h, w = img.shape[0], img.shape[1]
        y0 = (h - 224) // 2
        x0 = (w - 224) // 2
        img = img[y0: y0+224, x0: x0+224, :]
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        img = np.transpose(img, axes=[2, 0, 1])
        img = img.astype(np.float32)
        img = np.expand_dims(img, axis=0)
        return img

    def predict(self, image_data, post_process=False):
        inputs = {self.session.get_inputs()[0].name: image_data}

        start = timer()
        prediction = self.session.run(None, inputs)[0]
        inference_time = timer() - start

        if post_process:
            prediction = np.squeeze(prediction)
            a = np.argsort(prediction)[::-1]

            print('class = %s - probability = %f ' %
                  (self.classes[a[0]], prediction[a[0]]))

        return prediction, inference_time

class Emotion(BaseModel):
    # https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
    def __init__(self):
        BaseModel.__init__(self, 'emotion', 'model.onnx',
                           ModelFamily.COMPUTER_VISION)

        self.classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

    def prepare_dataset(self, path):
        pass

    def preprocess(self, image_path):
        '''
        input_shape = (1, 64, 64, 1)
        img = Image.open(image_path)
        img = img.resize((64, 64), Image.ANTIALIAS)
        img_data = np.array(img).astype(np.float32)
        img_data = np.reshape(img_data, input_shape)
        '''

        img = Image.open(image_path)
        img = np.asarray(img.resize((64, 64)))
        img = img / 255.0
        img = img - 0.5
        img = img * 2.0
        img = img.astype('float32')
        img_data = img.reshape((1, 64, 64, 1))

        dest_path = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/models/emotion/input_test.pb"
        tensor = numpy_helper.from_array(img_data)
        with open(dest_path, 'wb+') as f:
            f.write(tensor.SerializeToString())
        
        return img_data

    def softmax(self, x):
    
        y = np.exp(x - np.max(x))
        f_x = y / np.sum(np.exp(x))
        return f_x

    def predict(self, image_data, post_process=False):
        inputs = {self.session.get_inputs()[0].name: image_data}

        start = timer()
        prediction = self.session.run(None, inputs)[0]
        inference_time = timer() - start

        if post_process:
            print(inference_time, "(s)")
            out = np.array(prediction)
            prob = self.softmax(prediction)
            prob = np.squeeze(prob)
            pred = np.argsort(prob)[::-1]

            score = out[0].tolist()
            print(prediction)
            print(score)
            print("Emotion: \""+self.classes[score.index(max(score))]+"\" (" + str(max(score)*100) + " %)")

            #print(prediction)
            #print(out)

        return prediction, inference_time

class YOLOv4(BaseModel):
    def __init__(self, classes_file):
        BaseModel.__init__(self, 'yolov4', 'model.onnx',
                           ModelFamily.COMPUTER_VISION)

        self.classes = [l.rstrip() for l in open(classes_file, 'r')]

    def preprocess(self, image_path):

        img = cv2.imread('128.jpg')
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        dest_path = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/models/yolov4-128/input_test.pb"
        tensor = numpy_helper.from_array(img)
        with open(dest_path, 'wb+') as f:
            f.write(tensor.SerializeToString())

        return img

    def predict(self, image_data, post_process=False):
        inputs = {self.session.get_inputs()[0].name: image_data}

        start = timer()
        boxes, confs = self.session.run(None, inputs)
        inference_time = timer() - start

        if post_process:
            print(inference_time, '(s)')
            self.post_process(boxes, confs)
        
        return boxes, confs

    def post_process(self, boxes, confs):
        pass


class CIFAR10(BaseModel):
    # https://github.com/onnx/models/tree/master/vision/body_analysis/emotion_ferplus
    def __init__(self):
        BaseModel.__init__(self, 'cifar10', 'model.onnx',
                           ModelFamily.COMPUTER_VISION)

        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def prepare_dataset(self, path):
        pass

    def preprocess(self, image_path):
        pass

    def predict(self, image_data, post_process=False):
        inputs = {self.session.get_inputs()[0].name: image_data}

        start = timer()
        prediction = self.session.run(None, inputs)[0]
        inference_time = timer() - start

        if post_process:
            print(inference_time, "(s)")
            pass

        return prediction, inference_time

# MNIST


#print(onnxruntime.get_device())

'''
pb_path = "/home/ubuntu/rpi4/experiment/inputs/mnist/input_0.pb"
#pb_path = "/home/ubuntu/rpi4/experiment/inputs/mnist/pb_dataset/input_0_0.pb"
tensor = onnx.TensorProto()
with open(pb_path, 'rb') as f:
    tensor.ParseFromString(f.read())
img = numpy_helper.to_array(tensor)

mod = MNIST()
prediction, inference_time = mod.predict(img, post_process=True)
prediction, inference_time = mod.predict(img, post_process=True)
'''

'''
#pb_path = "/Users/krumielf/Desktop/Thesis/offline-app-pb/resources/cifar10/input_0.pb"
pb_path = "/home/ubuntu/rpi4/experiment/inputs/cifar10/input_0.pb"
tensor = onnx.TensorProto()
with open(pb_path, 'rb') as f:
    tensor.ParseFromString(f.read())

img_data = numpy_helper.to_array(tensor)
mod = CIFAR10()
pred = mod.predict(img_data, post_process=True)
pred = mod.predict(img_data, post_process=True)
#print(pred[0])
'''

'''
#pb_path = "/Users/krumielf/Desktop/Thesis/offline-app-pb/resources/yolov4_128/test_data_set_0/input_0.pb"
pb_path = "/home/ubuntu/rpi4/experiment/inputs/yolov4/input_0.pb"
tensor = onnx.TensorProto()
with open(pb_path, 'rb') as f:
    tensor.ParseFromString(f.read())

img_data = numpy_helper.to_array(tensor)
mod = YOLOv4(COCO_DATASET_CLASESS)
boxes, confs = mod.predict(img_data, post_process=True)
boxes, confs = mod.predict(img_data, post_process=True)
# print(boxes, confs)
'''

'''
#img_path = "/Users/krumielf/Desktop/Thesis/experiment/repository/experiment/models/emotion/data/happy/img_0.jpg"
pb_path = "/home/ubuntu/rpi4/experiment/inputs/emotion/input_0.pb"
tensor = onnx.TensorProto()
with open(pb_path, 'rb') as f:
    tensor.ParseFromString(f.read())
img = numpy_helper.to_array(tensor)
mod = Emotion()
mod.predict(img, post_process=True)
mod.predict(img, post_process=True)
'''

'''
img_path = "./inputs/image.png"
mod = MNIST()
img = mod.preprocess(img_path, from_dataset=False)
prediction, inference_time = mod.predict(img, post_process=True)
'''

'''
img_path = "./inputs/image.png"
mod = MNIST()
print()
print("Performing inference {} - {}".format(mod.name, mod.type))
#img = mod.preprocess(img_path)
images = mod.prepare_dataset()
count = 0
start = timer()

for j in range(60):
    #count += 1
    for i in range(len(images)):
        count += 1
        img = mod.preprocess(image=images[i], from_dataset=True)
        prediction, inference_time = mod.predict(img, post_process=True)
    #if count > 10:
    #    break

end = timer()
print(timedelta(seconds=end-start))
print("count:", count, "elapsed(s)", end - start)
'''

# Resnet

'''
img_path = "./inputs/kitten.jpg"
mod = Resnet(IMAGENET_DATASET_CLASSES)
image_list = mod.prepare_dataset("/Users/krumielf/Desktop/Thesis/imagenet_dataset/val")

#print()
#print("Performing inference {} - {} on {}".format(mod.name, mod.type, img_path))
count = 0
start = timer()
for img_path in image_list:
    count = count + 1
    img = mod.preprocess(img_path)
    print("Performing inference on {}".format(img_path))
    mod.predict(img, post_process=True)
    if count > 2500:
        break
end = timer()
print(timedelta(seconds=end-start))
print("count:", count, "elapsed(s)", end - start)
'''

'''
output_file = "/Users/krumielf/Desktop/Thesis/offline-app-pb/resources/cifar10/output_0.pb"
tensor = onnx.TensorProto()
with open(output_file, 'rb') as f:
    tensor.ParseFromString(f.read())
    i = numpy_helper.to_array(tensor)
    print (i)
'''