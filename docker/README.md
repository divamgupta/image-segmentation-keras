## Installing Docker

General installation instructions are
[on the Docker site](https://docs.docker.com/installation/), but we give some
quick links here:

* [OSX](https://docs.docker.com/installation/mac/): [docker toolbox](https://www.docker.com/toolbox)
* [ubuntu](https://docs.docker.com/installation/ubuntulinux/)

For GPU support, install compatible NVIDIA drivers with CUDA9.0 and CUDNN 7.6

## Running the container

Build the container:

    $ docker build -t isk

To run the image:

    $ docker run --gpus all -it isk

If you want to train with a dataset on your local machine, or make inference on images or videos, mount a volume to share this data with the docker container:

    $ docker run --gpus all -v /path/to/data/folder:/image-segmentation-keras/share -it isk

If graphical interface is needed, to show results, like `predict_video --display`, first let docker to use system interface. In your local host, type this line once:

    $ xhost +local:docker

And run the container with access to X11:

    $ docker run --gpus all -v /path/to/data/folder:/image-segmentation-keras/share -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix --env QT_X11_NO_MITSHM=1 -it isk
