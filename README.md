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

