import os

import numpy as np
import torch
import torch.nn as nn

mse = nn.MSELoss()
bce = nn.BCELoss()

def calculate_loss(model, batch, forcast_wt=2, device='cpu'):
    historical_seq_emb = batch['historical_seq_emb'].to(device)
    historical_seq_masked= batch['historical_seq_masked'].to(device)
    historical_seq = batch['historical_seq'].to(device)
    future_seq_emb = batch['future_seq_emb'].to(device)
    future_targets = batch['future_targets'].to(device)

    yh_reconstruction = model.encode_decode(historical_seq_emb, historical_seq_masked)
    yh_forecast = model.forecast(historical_seq_emb, historical_seq, future_seq_emb)

    loss_reconstruction = mse(yh_reconstruction, historical_seq)
    loss_forecast = bce(yh_forecast, future_targets)
    loss_joint = loss_reconstruction + forcast_wt*loss_forecast

    return loss_reconstruction, loss_forecast, loss_joint


def train_batch(model, optimizer, batch, device='cpu'):

    loss_reconstruction, loss_forecast, loss_joint = calculate_loss(model, batch, device=device)

    loss_joint.backward()
    optimizer.step()

    optimizer.zero_grad()
    
    return loss_reconstruction.item(), loss_forecast.item(), loss_joint.item()


def lr_schedule(n_steps, lr_min, lr_max):
    """Generates a concave learning rate schedule over
    `n_steps`, starting and ending at `lr_min` and
    peaking at `lr_max` mid-way.
    """
    schedule = np.arange(n_steps)/(n_steps - 1)
    schedule = np.sin(schedule*np.pi)
    schedule = lr_min + (lr_max - lr_min)*schedule
    return schedule


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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