# Use NVIDIA CUDA base image for GPU support
# Ensure this matches the CUDA version required by your PyTorch version
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update

RUN apt-get install -y --no-install-recommends \
      git \
      build-essential \
      software-properties-common \
      ca-certificates \
      wget \
      curl 

RUN apt-get update && apt-get install -y python3 python3-pip && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    pip3 install --upgrade pip
# Install system dependencies
# git & build-essential are required to compile mlperf_loadgen
#RUN apt-get update && apt-get install -y \
#    python3 \
#    python3-pip \
#    python3-dev \
#    git \
#    build-essential \
#    software-properties-common \
#    ca-certificates \
#    wget \
#    curl 
#    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies directly (removed dependency on requirements.txt)
RUN pip install torch>=2.1.0 \
    transformers>=4.40.0 \
    accelerate>=0.26.0 \
    datasets>=2.16.0 \
    numpy \
    tqdm \
    scikit-learn

# Install MLPerf LoadGen from source
# We clone the repo, build the python bindings, and then clean up the source to save space
RUN cd /tmp && \
    git clone --recursive https://github.com/mlcommons/mlperf_automotive && \
    cd mlperf_automotive/loadgen && \
    pip install pybind11 && \
    CFLAGS="-std=c++14" python setup.py install && \
    rm -rf mlperf

## Copy benchmark scripts
#COPY benchmark_runner.py .
#COPY download_data.py .

# Create directory for results
RUN mkdir -p results

# Set the default command to bash
CMD ["/bin/bash"]

