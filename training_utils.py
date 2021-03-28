import os

import numpy as np
import torch
import torch.nn as nn


def calculate_loss(model, batch, recon_wt=0.5, LCH_wt=1, mag_wt=0, dists_wt=1, device='cpu'):
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCHVT = batch['historical_seq_LCHVT'].to(device)
    h_LCH = h_LCHVT[:, :, :3]
    h_LCHVT_masked= batch['historical_seq_LCHVT_masked'].to(device)
    recon_loss_mask = batch['recon_loss_mask'].to(device)

    f_emb = batch['future_seq_emb'].to(device)
    f_LCH = batch['future_seq_LCH'].to(device)
    f_ret_dists = batch['future_ret_dists'].to(device)

    # Reconstruction loss
    yh_recon = model.encode_decode(h_emb, h_LCHVT_masked)
    loss_recon = torch.abs(yh_recon - h_LCH)
    loss_recon = torch.sum(loss_recon*recon_loss_mask)/torch.sum(recon_loss_mask)/3

    # Future return path and distribution losses
    yh_LCH, yh_ret_probas = model.forecast(h_emb, h_LCHVT, f_emb)
    loss_LCH = torch.mean(torch.mean(torch.abs(yh_LCH - f_LCH), dim=(1,2)))
    loss_mag = torch.mean(torch.mean(torch.abs(torch.abs(yh_LCH) - torch.abs(f_LCH)), dim=(1,2)))
    loss_dists = nn.BCELoss()(yh_ret_probas, f_ret_dists)

    # Scale and combine losses
    loss = (
        loss_recon*recon_wt + 
        loss_LCH*LCH_wt + 
        loss_mag*mag_wt +
        loss_dists*dists_wt
    )/4

    return loss_recon.item(), loss_LCH.item(), loss_mag.item(), loss_dists.item(), loss


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