FROM python:3.9-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         sudo \
         procps \
         sysstat \
         git \
         vim \
         libprotoc-dev \
         protobuf-compiler \
         python3-pip \
         python3-opencv \
         libgl1-mesa-glx

RUN pip3 install onnx \
        onnxruntime \
        numpy \
        matplotlib \
        idx2numpy \
        opencv-python 

#WORKDIR /home/experiment

#ENTRYPOINT ["/bin/bash"]
CMD [ "python3", "./measure.py" ]