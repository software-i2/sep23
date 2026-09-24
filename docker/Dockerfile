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
        tmux \
        libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Every interactive shell, tmux panes included, gets ROS and the workspace named by $WS.
RUN echo 'source /opt/ros/noetic/setup.bash; [ -f "$WS/devel/setup.bash" ] && source "$WS/devel/setup.bash"' >> /root/.bashrc
