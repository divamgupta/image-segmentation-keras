ARG CUDA="9.0"
ARG CUDNN="7"

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu16.04
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# install prerequisites
RUN apt-get update -y \
 && apt-get upgrade -y \
 && apt-get install nano \
 && apt-get install curl \
 && apt install -y git \
 && apt-get install -y libsm6 libxext6 libxrender-dev

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

# Create a Python 3.6 environment
ENV PATH=/miniconda/bin:$PATH

RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name unet python=3.6.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=unet
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython tensorflow-gpu keras opencv

# install model library
RUN git clone https://github.com/divamgupta/image-segmentation-keras.git
WORKDIR /image-segmentation-keras
RUN python setup.py install