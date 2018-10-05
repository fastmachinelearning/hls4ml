FROM continuumio/miniconda
MAINTAINER Vladimir Loncar

RUN conda update conda

# Setup Python 3.6 environment
RUN conda create --copy --name hls4ml-py36 python=3.6 && \
    conda install pytorch-cpu torchvision-cpu -c pytorch --name hls4ml-py36 -y && \
    conda install -c anaconda keras scikit-learn h5py pyyaml --name hls4ml-py36 -y

# Setup Python 2.7 environment
RUN conda create --copy --name hls4ml-py27 python=2.7 && \
    conda install pytorch-cpu torchvision-cpu -c pytorch --name hls4ml-py27 -y && \
    conda install -c anaconda keras scikit-learn h5py pyyaml --name hls4ml-py27 -y

RUN apt-get update && apt-get install -y \
  build-essential \
  git \
  libglib2.0-0 \
  libsm6 \
  libxi6 \
  libxrender1 \
  libxrandr2 \
  libfreetype6 \
  libfontconfig \
  lsb-release \
  procps

# Copy Vivado installation folder
COPY install_config.txt /tmp/

ADD Xilinx_Vivado_SDK_2017.2_0616_1 /tmp/Xilinx_Vivado_SDK_2017.2_0616_1
RUN /tmp/Xilinx_Vivado_SDK_2017.2_0616_1/xsetup --agree 3rdPartyEULA,WebTalkTerms,XilinxEULA --batch Install -c /tmp/install_config.txt

#ADD Xilinx_Vivado_SDK_2018.2_0614_1954 /tmp/Xilinx_Vivado_SDK_2018.2_0614_1954
#RUN /tmp/Xilinx_Vivado_SDK_2018.2_0614_1954/xsetup --agree 3rdPartyEULA,WebTalkTerms,XilinxEULA --batch Install -c /tmp/install_config.txt

RUN rm -rf /tmp/*

ARG LICENSE_SERVER
ENV XILINXD_LICENSE_FILE $LICENSE_SERVER
