
# Base image - CUDA enabaled PyTorch
FROM nvcr.io/nvidia/pytorch:21.04-py3

# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

# Set the working directory in the container
WORKDIR /opt/ml/code

# Install dependencies
COPY ./containers/training/requirements.txt requirements.txt
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt

# Copy code to container's wd
COPY data_utils.py data_utils.py
COPY modules.py modules.py
COPY training_utils.py training_utils.py
COPY ./containers/training/train.py train.py

# Define train.py as script entrypoint
ENTRYPOINT ["python", "train.py"]