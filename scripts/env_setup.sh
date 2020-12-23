#!/bin/bash
conda create -y -n neuralopt_env python=3.8 pip
source activate neuralopt_env
python -m pip install -r requirements.txt
python -m ipykernel install --user --name neuralopt_env
