FROM tensorflow/tensorflow:2.0.0-gpu-py3

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install tzdata

RUN pip install --upgrade pip && pip install \
  tqdm \
  scipy \
  pydensecrf \
  tensorflow-addons \
  pandas

WORKDIR /src
