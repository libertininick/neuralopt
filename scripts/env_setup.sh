#!/bin/bash
conda update -n base conda
conda create -y -n neuralopt_env pip

source activate neuralopt_env

python -m pip install -r requirements.txt
conda install -y -c conda-forge ipykernel ipywidgets==7.5 jupyter jupyter_contrib_nbextensions jupytext nb_conda 
conda install -y -c anaconda notebook>=5.3 
conda install -y -c plotly plotly==4.11.0