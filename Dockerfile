# If using GPU replace the following line with:
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04
FROM ubuntu:20.04

RUN apt-get update && apt-get upgrade -y && apt-get install python3-dev python3-pip -y
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN mkdir /sova-nlu
WORKDIR /sova-nlu
