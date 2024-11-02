#! /bin/bash

# Specify the Python version
PYTHON_VERSION=3.10

# Install the specified Python version
sudo apt-get update
sudo apt-get install -y python${PYTHON_VERSION} python3-pip
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
sudo apt-get install -y python3-venv
