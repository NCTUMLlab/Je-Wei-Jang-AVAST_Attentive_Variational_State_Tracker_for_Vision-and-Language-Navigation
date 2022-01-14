# Matterport3DSimulator
# Requires nvidia gpu with driver 396.37 or higher
FROM nvidia/cudagl:11.2.2-devel-ubuntu18.04

# Install a few libraries to support both EGL and OSMESA options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y wget doxygen curl apt-utils libjsoncpp-dev libepoxy-dev libglm-dev libosmesa6 libosmesa6-dev libglew-dev libopencv-dev python-opencv python3-setuptools python3-dev python3-pip
RUN pip3 install opencv-python==4.1.0.25 numpy==1.13.3 pandas==0.24.1 networkx==2.2

#install latest cmake
ADD https://cmake.org/files/v3.12/cmake-3.12.2-Linux-x86_64.sh /cmake-3.12.2-Linux-x86_64.sh
RUN mkdir /opt/cmake
RUN sh /cmake-3.12.2-Linux-x86_64.sh --prefix=/opt/cmake --skip-license
RUN ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake
RUN cmake --version


#############
# Customize #
#############

# set noninteractive installation
RUN echo export DEBIAN_FRONTEND=noninteractive
#install tzdata package
RUN apt-get update && apt-get install -y \
    tzdata \
 && rm -rf /var/lib/apt/lists/*
# set your timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata


RUN apt-get update && apt-get install -qqy \
    x11-apps \
    locales \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    openssh-server \
    vim	\
    ffmpeg \
    htop \
    python3-tk \
 && rm -rf /var/lib/apt/lists/*

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN pip3 install matplotlib==3.3.4 nltk==3.6.2 tqdm==4.61.0 unidecode==1.2.0 tensorflow==1.14.0 tensorboardX==2.1 moviepy==1.0.3 flake8==3.9.2 flake8-unused-arguments==0.0.6
RUN pip3 install torch

# Locale
RUN locale-gen en_US.UTF-8
RUN locale-gen zh_TW.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8


ENV PYTHONPATH=/root/mount/Matterport3DSimulator/build

CMD /bin/sh -c 'service ssh restart && bash'
