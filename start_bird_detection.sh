#!/bin/bash

# Exit immediately if any command fails
set -e

# (Optional) Go to your project directory
cd /home/wombach/github

# Activate virtual environment
source ./birdenv/bin/activate

cd bits_pieces/jonathan/03_raspberry/

# Run your Python job
python bird_cam_v2.py

# (Optional) Deactivate when done
deactivate
