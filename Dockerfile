FROM nvcr.io/nvidia/deepstream:6.4-samples-multiarch

# To get video driver libraries at runtime
ENV NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES,video

# General update and install nala
RUN apt update && \
    apt install -y nala wget

# Locale fixing
RUN nala install -y locales && \
    locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ROS2 installation
RUN nala install -y software-properties-common && \
    add-apt-repository universe && \
    nala update && nala install -y curl lsb-release gnupg && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" > /etc/apt/sources.list.d/ros2.list && \
    nala update && nala install -y ros-humble-desktop

# Gazebo installation
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list && \
    nala update && nala install -y gz-harmonic

# Sourcing ROS2 (for runtime)
ENV ROS2_SOURCE_CMD="source /opt/ros/humble/setup.bash"

# DeepStream Python bindings setup
RUN cd /home/ && \
    git clone -b v1.1.10 --recurse-submodules https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
    wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.10/pyds-1.1.10-py3-none-linux_x86_64.whl && \
    nala install python3-pip -y 

RUN echo '#!/bin/bash\n\
pip install /home/pyds-1.1.10-py3-none*.whl' > /home/pip_install.sh

# Make the bash script executable and run it
RUN chmod +x /home/pip_install.sh && sh /home/pip_install.sh

# installing gstreamer 
RUN nala install \
libssl3 \
libssl-dev \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstreamer-plugins-base1.0-dev \
libgstrtspserver-1.0-0 \
libjansson4 \
libyaml-cpp-dev -y

# deepstream python apps to samples directory in deepstream 
RUN cp -r /home/deepstream_python_apps/ /opt/nvidia/deepstream/deepstream/samples/.
