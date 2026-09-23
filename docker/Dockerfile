FROM ros:noetic-ros-base

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-catkin-tools \
        ros-noetic-foxglove-bridge \
        ros-noetic-kdl-parser \
        ros-noetic-ompl \
        ros-noetic-robot-state-publisher \
        ros-noetic-tf2-eigen \
        ros-noetic-tf2-ros \
        ros-noetic-urdf \
        ros-noetic-xacro \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

RUN echo 'source /opt/ros/noetic/setup.bash' >> /root/.bashrc
