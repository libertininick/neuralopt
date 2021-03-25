import os

import numpy as np
import torch
import torch.nn as nn


bce = nn.BCELoss()

def calculate_loss(model, batch, recon_wt=1/0.667, path_wt=1/0.831, mag_wt=1/0.517, dists_wt=1/0.455, device='cpu'):
    historical_seq_emb = batch['historical_seq_emb'].to(device)
    historical_seq = batch['historical_seq'].to(device)
    historical_seq_masked= batch['historical_seq_masked'].to(device)
    recon_loss_mask = batch['loss_mask'].to(device)

    future_seq_emb = batch['future_seq_emb'].to(device)
    future_ret_path = batch['future_ret_path'].to(device)

    batch_size, seq_len = future_ret_path.shape
    future_ret_mag = torch.abs(torch.diff(
        future_ret_path*torch.arange(seq_len).unsqueeze(0).to(device)**0.5, 
        dim=1, 
        prepend=torch.tensor([[0.0]]*batch_size, device=device)
    ))
    future_ret_dists = batch['future_ret_dists'].to(device)

    # Reconstruction loss
    yh_recon = model.encode_decode(historical_seq_emb, historical_seq_masked)
    loss_recon = torch.abs(yh_recon - historical_seq[:, :, :3])
    loss_recon = torch.sum(loss_recon*recon_loss_mask)/torch.sum(recon_loss_mask)/3

    # Future return path and distribution losses
    yh_path, yh_probas = model.forecast(historical_seq_emb, historical_seq, future_seq_emb)
    yh_mags = torch.abs(torch.diff(
        yh_path*torch.arange(seq_len).unsqueeze(0).to(device)**0.5, 
        dim=1, 
        prepend=torch.tensor([[0.0]]*batch_size, device=device)
    ))
    loss_path = torch.mean(torch.mean(torch.abs(yh_path - future_ret_path), dim=1))
    loss_mag = torch.mean(torch.mean(torch.abs(yh_mags - future_ret_mag), dim=1))
    loss_dists = bce(yh_probas, future_ret_dists)

    # Scale and combine losses
    loss = (
        loss_recon*recon_wt + 
        loss_path*path_wt + 
        loss_mag*mag_wt +
        loss_dists*dists_wt
    )/4

    return loss_recon.item(), loss_path.item(), loss_mag.item(), loss_dists.item(), loss


def train_batch(model, optimizer, batch, device='cpu'):

    loss_recon, loss_path, loss_mag, loss_dists, loss = calculate_loss(model, batch, device=device)

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    
    return loss_recon, loss_path, loss_mag, loss_dists, loss.item()


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