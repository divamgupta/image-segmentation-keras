FROM tanmaniac/opencv3-cudagl

# install prerequisites
RUN apt-get update \
 && apt-get install -y wget git curl nano \
 && apt-get install -y libsm6 libxext6 libxrender-dev

# install Cudnn
ENV CUDNN_VERSION 7.6.0.64
RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh \
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

RUN conda install -y ipython tensorflow-gpu=1.14.0 keras=2.3.1

# install model library
RUN git clone https://github.com/divamgupta/image-segmentation-keras.git
WORKDIR /image-segmentation-keras
RUN python setup.py install
