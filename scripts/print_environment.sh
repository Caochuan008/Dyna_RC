#!/usr/bin/env bash
# Run on the Ubuntu machine used for building/evaluation. Does not install anything.
set -u
printf 'OS\n'
cat /etc/os-release
printf '\nCompiler / CMake / Python\n'
g++ --version | head -n 1
cmake --version | head -n 1
python3 --version
printf '\nROS distribution and roscpp\n'
rosversion -d
rosversion roscpp
printf '\nInstalled dependency versions\n'
dpkg-query -W -f='${Package}\t${Version}\n' \
  libpcl-dev libeigen3-dev libopencv-dev libboost-dev \
  python3-numpy python3-scipy python3-matplotlib \
  ros-noetic-catkin ros-noetic-pcl-ros ros-noetic-pcl-conversions 2>/dev/null || true
printf '\nPython packages actually imported\n'
python3 - <<'PY'
import importlib
for name in ('numpy', 'scipy', 'matplotlib'):
    try:
        module = importlib.import_module(name)
        print(name, module.__version__, module.__file__)
    except ImportError as error:
        print(name, error)
PY
