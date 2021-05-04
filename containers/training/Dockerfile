
# Base image: AWS DLC PyTorch 1.8.1 GPU training image
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04

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