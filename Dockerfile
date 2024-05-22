FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# install required packages
RUN apt-get update && apt-get install -y iproute2

# get source from git
COPY src /workspace/src
COPY *.py /workspace/
RUN mkdir -p /workspace/data/dat/train
RUN mkdir -p /workspace/data/dat/test

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
ENV CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log

# Mount volumes
VOLUME /lib/modules
VOLUME /tmp/nvidia-mps

COPY scripts/tc-script.sh /usr/local/bin/tc-script.sh

RUN chmod +x /usr/local/bin/tc-script.sh

CMD ["/bin/sh", "-c", "/usr/local/bin/tc-script.sh && sleep infinity"]
