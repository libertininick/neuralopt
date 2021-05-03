# Containers

## Training Images
These images are for running training jobs on AWS Sagemaker

### Featurizer
Build the image:
```bash
docker build \
--rm \
--tag=price-series/featurizer/training:1.0.0 \
--file=./containers/training/Dockerfile \
.
```

Test the image locally. For model training, Amazon SageMaker runs the container as follows: `docker run <image name> train`.
```bash
docker run \
--rm \
--name=training-test \
--mount type=bind,source=/path/to/local/input,destination=/opt/ml/input/data/all,readonly \
--mount type=bind,source=/path/to/local/model,destination=/opt/ml/model \
--mount type=bind,source=/path/to/local/output,destination=/opt/ml/output \
price-series/featurizer/training:1.0.0 train
```

Tag and Push to AWS ECR:
```bash
# Pipe AWS credentials to docker
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account id>.dkr.ecr.<region>.amazonaws.com

# Tag image
docker tag <image id> <account id>.dkr.ecr.<region>.amazonaws.com/<image name>:<tag>

# Push to AWS ECR
docker push <account id>.dkr.ecr.<region>.amazonaws.com/<image name>:<tag>
```