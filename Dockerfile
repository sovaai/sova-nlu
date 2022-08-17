# If using GPU replace the following line with:
# FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM python

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

RUN mkdir /sova-nlu
WORKDIR /sova-nlu
