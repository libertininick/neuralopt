import os

import numpy as np
import torch
import torch.nn as nn


def calculate_feat_loss(model, batch, recon_wt=0.5, seq_order_wt=0.5, LCH_wt=1, dists_wt=1, device='cpu'):
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    h_LCH_masked = batch['historical_seq_LCH_masked'].to(device)
    recon_loss_mask = batch['recon_loss_mask'].to(device)

    f_emb = batch['future_seq_emb'].to(device)
    f_LCH = batch['future_seq_LCH'].to(device)
    f_ret_dists = batch['future_ret_dists'].to(device)

    # Reconstruction loss
    yh_recon = model.encode_decode(h_emb, h_LCH_masked)
    loss_recon = torch.abs(yh_recon - h_LCH)
    loss_recon = torch.sum(loss_recon*recon_loss_mask)/torch.sum(recon_loss_mask)/3

    # Sequence order prediction
    batch_size, h_seq_len = h_LCH.shape[:2]
    seq_len = 32
    buffer = np.random.randint(low=0, high=seq_len)
    st_idx = np.random.randint(low=0, high=h_seq_len - seq_len*2 - buffer)
    end_idx = st_idx + seq_len*2 + buffer
    x_emb, x_LCH = h_emb[:,st_idx:end_idx,:], h_LCH[:,st_idx:end_idx,:]
    if np.random.rand() < 0.5:
        # Correct ordering
        x_emb_a, x_LCH_a = x_emb[:,:seq_len,:], x_LCH[:,:seq_len,:]
        x_emb_b, x_LCH_b = x_emb[:,seq_len + buffer:,:], x_LCH[:,seq_len + buffer:,:]

        order_targets = torch.ones(size=(batch_size,1), dtype=torch.float, device=device)
    else:
        # Swapped ordering
        x_emb_a, x_LCH_a = x_emb[:,seq_len + buffer:,:], x_LCH[:,seq_len + buffer:,:]
        x_emb_b, x_LCH_b = x_emb[:,:seq_len,:], x_LCH[:,:seq_len,:]
        order_targets = torch.zeros(size=(batch_size,1), dtype=torch.float, device=device)

    seq_a = model.vectorize_seq(x_emb_a, x_LCH_a)
    seq_b = model.vectorize_seq(x_emb_b, x_LCH_b)
    yh_seq_order = model.seq_order_classifier(torch.cat((seq_a, seq_b), dim=-1))
    loss_seq_order = nn.BCELoss()(yh_seq_order, order_targets)


    # Future return path and distribution losses
    yh_LCH, yh_ret_probas = model.forecast(h_emb, h_LCH, f_emb)
    loss_LCH = torch.mean(torch.mean(torch.abs(yh_LCH - f_LCH), dim=(1,2)))
    loss_dists = nn.BCELoss()(yh_ret_probas, f_ret_dists)

    # Scale and combine losses
    loss = (
        loss_recon*recon_wt + 
        loss_seq_order*seq_order_wt +
        loss_LCH*LCH_wt + 
        loss_dists*dists_wt
    )

    return loss_recon.item(), loss_seq_order.item(), loss_LCH.item(), loss_dists.item(), loss


def train_feat_batch(model, optimizer, batch, device='cpu'):

    loss_recon, loss_seq_order, loss_path, loss_dists, loss = calculate_feat_loss(model, batch, device=device)

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    
    return loss_recon, loss_seq_order, loss_path, loss_dists, loss.item()


def _wasserstein_gradients(critic, context, real, fake, epsilon):
    """Gradient of the critic's scores with respect to mixes of real and fake examples.
    
    Args:
        critic (PriceSeriesCritic): Critic model
        context (tensor): Context of each example (batch_size, seq_len, n_features)
        real (tensor): a batch of real examples (batch_size, seq_len, 3)
        fake  (tensor): a batch of fake examples (batch_size, seq_len, 3)
        epsilon (tensor): uniformly random proportions of real/fake per mixed example (batch_size, 1, 1)
    
    Returns:
        gradients (tensor): Gradient of critic's scores, with respect to the mixed examples (batch_size, seq_len, 3)
    """
    # Mix examples together
    mixed = real*epsilon + fake*(1 - epsilon)

    # Calculate the critic's scores on the mixed examples
    scores = critic(context, mixed)
    
    # Take the gradient of the scores with respect to the mixed inputs
    gradients = torch.autograd.grad(
        inputs=mixed,
        outputs=scores,
        grad_outputs=torch.ones_like(scores), 
        create_graph=True,
        retain_graph=True,
    )[0]

    return gradients


def _wasserstein_penalty(gradients):
    """Wasserstein gradient penalty, given a batch of gradients.

    Args:
        gradients (tensor): the gradient of the critic's scores, with respect to the mixed examples (batch_size, seq_len, 3)
    
    Returns:
        penalty (tensor): Scalar penalty
    """
    # Flatten the gradients so that each row captures one example
    gradients = gradients.reshape(len(gradients), -1)

    # Calculate the magnitude of every example
    gradient_norms = gradients.norm(2, dim=1)
    
    # Penalize the mean squared distance of the gradient norms from 1
    penalty = torch.mean((gradient_norms - 1)**2)

    return penalty


def calculate_critic_loss(critic, context, real, fake, penalty_wt, device='cpu'):
    """Critic's loss given the a batch of real and fake examples
    
    Args:
        critic (PriceSeriesCritic): Critic model
        context (tensor): Context of each example (batch_size, seq_len, n_features)
        real (tensor): a batch of real examples (batch_size, seq_len, 3)
        fake  (tensor): a batch of fake examples (batch_size, seq_len, 3)
        penalty_wt (float): the current weight of the gradient penalty 
    
    Returns:
        loss: a scalar for the critic's loss
    """
    batch_size = context.shape[0]

    # Scores
    score_real = torch.mean(torch.mean(critic(context, real), dim=-1))
    score_fake = torch.mean(torch.mean(critic(context, fake), dim=-1))

    # wasserstein penalty
    wass_grads = _wasserstein_gradients(
        critic, 
        context, 
        real, 
        fake, 
        epsilon=torch.rand(
            size=(batch_size,1,1), 
            device=device, 
            requires_grad=True
        )  # Uniformly random proportions of real/fake per mixed example
    )
    grad_penalty = _wasserstein_penalty(wass_grads)

    loss = -(score_real - score_fake) + penalty_wt*grad_penalty
    
    return score_real.item(), score_fake.item(), loss


def train_GC_wass_batch(
    models, 
    optimizers, 
    batch, 
    critic_repeats=5, 
    wass_penalty_wt=1, 
    device='cpu'):

    featurizer, generator, critic = models
    gen_opt, critic_opt = optimizers

    # Batch data
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    f_emb = batch['future_seq_emb'].to(device)
    LCH_real = batch['future_seq_LCH'].to(device)
    batch_size = h_emb.shape[0]
    

    ### Get context ###
    with torch.no_grad():
        context = featurizer.encode_future_path(h_emb, h_LCH, f_emb)


    ### Update critic ###
    scores_real, scores_fake = [], []
    for _ in range(critic_repeats):

        # Generate fake examples
        noise = generator.gen_noise(batch_size, device=device)
        with torch.no_grad():
            LCH_fake = generator(context, noise)
        
        # Critic loss
        score_real, score_fake, critic_loss = calculate_critic_loss(
            critic,
            context,
            LCH_real, 
            LCH_fake, 
            wass_penalty_wt
        )
        scores_real.append(score_real)
        scores_fake.append(score_fake)
        
        # Backprop
        critic_loss.backward(retain_graph=True)
        
        # Update critic's weights
        critic_opt.step()
        critic_opt.zero_grad()

    
    ### Update generator ###
    noise = generator.gen_noise(batch_size, device=device)
    LCH_fake = generator(context, noise)
    score_fake = torch.mean(torch.mean(critic(context, LCH_fake), dim=-1))
    gen_loss = -score_fake
    
    # Backprop
    gen_loss.backward()

    # Update generator's weights
    gen_opt.step()
    gen_opt.zero_grad()

    return np.mean(scores_real), np.mean(scores_fake)


def train_GC_least_square_batch(
    models, 
    optimizers, 
    batch, 
    critic_repeats=5, 
    device='cpu'):

    featurizer, generator, critic = models
    gen_opt, critic_opt = optimizers

    # Batch data
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    f_emb = batch['future_seq_emb'].to(device)
    LCH_real = batch['future_seq_LCH'].to(device)
    batch_size = h_emb.shape[0]
    

    ### Get context ###
    with torch.no_grad():
        context = featurizer.encode_future_path(h_emb, h_LCH, f_emb)


    ### Update critic ###
    critic_losses = []
    for _ in range(critic_repeats):

        # Generate fake examples
        with torch.no_grad():
            noise = generator.gen_noise(batch_size, device=device)
            LCH_fake = generator(context, noise)
        
        # Calculate scores
        scores_real = critic(context, LCH_real)
        scores_fake = critic(context, LCH_fake)

        # Critic loss
        loss_real = torch.mean(torch.mean((scores_real - 1)**2, dim=-1))
        loss_fake = torch.mean(torch.mean((scores_fake - 0)**2, dim=-1))
        critic_loss = (loss_real + loss_fake)/2
        critic_losses.append(critic_loss.item())
        
        # Backprop
        critic_loss.backward(retain_graph=True)
        
        # Update critic's weights
        critic_opt.step()
        critic_opt.zero_grad()

    
    ### Update generator ###
    noise = generator.gen_noise(batch_size, device=device)
    LCH_fake = generator(context, noise)
    scores_fake = critic(context, LCH_fake)
    gen_loss = torch.mean(torch.mean((scores_fake - 1)**2, dim=-1))
    
    # Backprop
    gen_loss.backward()

    # Update generator's weights
    gen_opt.step()
    gen_opt.zero_grad()

    return gen_loss.item()**0.5, np.mean(critic_losses)**0.5


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