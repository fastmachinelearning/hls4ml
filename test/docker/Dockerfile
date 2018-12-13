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

# Install necessary packages
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
    procps \
    sudo

# Multilib support (workaround required by some configurations)
RUN apt-get install -y \
    gcc-multilib \
    g++-multilib && \
    ln -s /usr/lib/x86_64-linux-gnu /usr/lib64

# Copy Vivado installation folder
COPY install_config.txt /tmp/

#ADD Xilinx_Vivado_SDK_2017.2_0616_1 /tmp/Xilinx_Vivado_SDK_2017.2_0616_1
#RUN /tmp/Xilinx_Vivado_SDK_2017.2_0616_1/xsetup --agree 3rdPartyEULA,WebTalkTerms,XilinxEULA --batch Install -c /tmp/install_config.txt && \
#   rm -rf /tmp/*

ADD Xilinx_Vivado_SDK_2018.2_0614_1954 /tmp/Xilinx_Vivado_SDK_2018.2_0614_1954
RUN /tmp/Xilinx_Vivado_SDK_2018.2_0614_1954/xsetup --agree 3rdPartyEULA,WebTalkTerms,XilinxEULA --batch Install -c /tmp/install_config.txt && \
    rm -rf /tmp/*

# Setup license server
ARG LICENSE_SERVER
ENV XILINXD_LICENSE_FILE $LICENSE_SERVER

# Install packages required for running Vivado HLS GUI
ARG GUI_SUPPORT
RUN if [ "$GUI_SUPPORT" = "1" ] ; then \
      export DEBIAN_FRONTEND=noninteractive && \
      apt-get install -y \
      default-jre \
      xorg; \
    fi

# Setup default user
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN mkdir -p /home/hls4ml && \
    echo "hls4ml:x:${USER_ID}:${GROUP_ID}:hls4ml User,,,:/home/hls4ml:/bin/bash" >> /etc/passwd && \
    echo "hls4ml:x:${USER_ID}:" >> /etc/group && \
    echo "hls4ml ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/hls4ml && \
    chmod 0440 /etc/sudoers.d/hls4ml && \
    chown ${USER_ID}:${GROUP_ID} -R /home/hls4ml

USER hls4ml
ENV HOME /home/hls4ml
WORKDIR /home/hls4ml
# Note that this may fail if there are multiple Vivado installations at the same path
RUN cp /etc/skel/.bashrc .bashrc && \
    echo "source /opt/Xilinx/Vivado/*/settings64.sh" >> .bashrc && \
    echo "source activate hls4ml-py36" >> .bashrc && \
    git clone https://github.com/hls-fpga-machine-learning/hls4ml.git
