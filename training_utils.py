import os

import torch
import torch.nn as nn


def save_state_dict_to_s3(model, file_name, bucket_name, subfolder=None):
    import boto3
    
    s3_client = boto3.client('s3')
    
    # Save state_dict in current working directory
    torch.save(model.state_dict(), file_name)
    
    # Upload to S3
    if subfolder:
        object_name = f'{subfolder}/{file_name}'
    else:
        object_name = file_name
        
    response = s3_client.upload_file(file_name, bucket_name, object_name)
    
    # Remove local copy
    os.remove(file_name)
    
    return response