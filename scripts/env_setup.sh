#!/bin/bash 

# Create a conda environment named `neuralopt_env` 
conda create -y -n neuralopt_env python=3.7 pip 

# Activate environment 
source activate neuralopt_env

# pip install the required Python packages 
python -m pip install -r requirements.txt 

# Optional 
# Install Jupyter notebook, et al. 
conda install -y -c conda-forge notebook                      # classic Jupyter notebook 
conda install -y -c conda-forge nb_conda                      # kernels for conda environments 
conda install -y -c conda-forge jupyter_contrib_nbextensions  # notebook extensions
conda install -y -c conda-forge jupytext                      # Translates notebooks to lite Python scripts 

# Add notebook extensions 
jupyter nbextension enable collapsible_headings/main  # Collapsible sections
jupyter nbextension enable execute_time/ExecuteTime   # Display cell execution time
jupyter nbextension enable hinterland/hinterland      # Enable code autocompletion
jupyter nbextension enable spellchecker/main          # Spellchecking 
jupyter nbextension enable toc2/main                  # TOC for headers as a sidebar
