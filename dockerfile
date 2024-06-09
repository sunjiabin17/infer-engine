FROM nvcr.io/nvidia/cuda:12.3.2-devel-ubuntu22.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    gdb \
    vim \
    python3 \
    python3-pip

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgoogle-glog-dev \
    libgtest-dev \
    libprotobuf-dev \
    protobuf-compiler

    