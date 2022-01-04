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


ENV GOLANG_VERSION 1.15.7

# Install wget
RUN apt-get update && apt-get install -y build-essential wget
# Install Go
RUN wget https://golang.org/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz
RUN tar -C /usr/local -xzf go${GOLANG_VERSION}.linux-amd64.tar.gz
RUN rm -f go${GOLANG_VERSION}.linux-amd64.tar.gz

ENV PATH "$PATH:/usr/local/go/bin"

# Tensorboard ports
EXPOSE 6006