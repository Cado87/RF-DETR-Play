# RF-DETR Roboflow
Tests of RF-DETR model
Sources: https://rfdetr.roboflow.com/learn/pretrained/

First check if rf-detr-env exists:
conda env list

If doesn´t exist, create a new Conda environment. Open your terminal and run the following command:
conda create --name rf-detr-env python=3.11 -y

Activate the new environment:
conda activate rf-detr-env

# Use the default base model
python Run-Pre-Trained-Model.py

# Use the nano model (fastest)
python Run-Pre-Trained-Model.py --model nano

# Use the small model
python Run-Pre-Trained-Model.py --model small

# Use the medium model
python Run-Pre-Trained-Model.py --model medium

# Use the large model (most accurate)
python Run-Pre-Trained-Model.py --model large

## Webcam Options

# Use default webcam (camera index 0)
python Run-Pre-Trained-Model.py --camera 0

# Use USB webcam (camera index 1)
python Run-Pre-Trained-Model.py --camera 1

# Combine model selection with camera selection
python Run-Pre-Trained-Model.py --model nano --camera 1

# Load custom weights (auto-detects num_classes and class names from checkpoint)
python Run-Pre-Trained-Model.py --model nano --checkpoint Weights/checkpoint_best_total_model_nano.pth

# Load custom weights with manual num_classes specification (if auto-detection fails)
python Run-Pre-Trained-Model.py --model nano --checkpoint Weights/checkpoint_best_total_model_nano.pth --num-classes 8

