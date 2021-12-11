FROM ubuntu:21.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update

RUN apt-get install -y \
	libopencv-dev \
        python3-pip \
	python3-opencv && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install tensorflow && \
    pip3 install numpy \
        pandas \
        sklearn \
        matplotlib \
        jupyter \
        nltk && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils


RUN apt install -y golang

# Tensorboard ports
EXPOSE 6006