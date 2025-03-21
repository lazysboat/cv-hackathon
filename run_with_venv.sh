#!/bin/bash

echo "AI vs Real Image Detector"
echo "========================"
echo

# Check if virtual environment exists
if [ -d "ai_detector_env" ]; then
    echo "Virtual environment found."
else
    echo "Creating virtual environment..."
    python3 setup_venv.py
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
fi

echo
echo "Activating virtual environment..."
source ai_detector_env/bin/activate

if [ $? -ne 0 ]; then
    echo "Failed to activate virtual environment."
    exit 1
fi

echo
echo "Virtual environment activated successfully!"
echo
echo "Available commands:"
echo "- python ai_detector.py [image_path]"
echo "- python batch_detector.py [directory_path]"
echo "- python gui_detector.py"
echo
echo "Type 'deactivate' when finished to exit the virtual environment."
echo

# Keep the shell open
exec $SHELL 