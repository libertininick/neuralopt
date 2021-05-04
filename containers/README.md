# Containers

## Training Images
These images are for running training jobs on AWS Sagemaker

### Featurizer
Download base image from AWS:
```bash
#  Login to access to the AWS DLC image repository
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com

# Pull PyTorch 1.8.1 GPU image
docker pull 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.8.1-gpu-py36-cu111-ubuntu18.04
```

Build the image:
```bash
docker build \
--rm \
--tag=price-series/featurizer/training:1.0.0 \
--file=./containers/training/Dockerfile \
.
```

Test the image locally:
```bash
docker run \
--rm \
--name=training-test \
--mount type=bind,source=/path/to/local/input,destination=/opt/ml/input/data/all,readonly \
--mount type=bind,source=/path/to/local/model,destination=/opt/ml/model \
--mount type=bind,source=/path/to/local/output,destination=/opt/ml/output \
price-series/featurizer/training:1.0.0
```
Note, Amazon SageMaker passes either `train` or `serve` when running an image. For model training, SageMaker runs the container as follows: `docker run <image name> train`. The `train` argument will be handled by argparse's `.parse_known_args()` method. 

Tag and Push to AWS ECR:
```bash
# Pipe AWS credentials to docker
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account id>.dkr.ecr.<region>.amazonaws.com

# Tag image
docker tag <image id> <account id>.dkr.ecr.<region>.amazonaws.com/<image name>:<tag>

# Push to AWS ECR
docker push <account id>.dkr.ecr.<region>.amazonaws.com/<image name>:<tag>
```