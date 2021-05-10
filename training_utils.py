import os

try:
    import boto3
except:
    print('boto3 library unavailable')
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn

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


def sample_contrastive_pairs(batch, featurizer, sample_len, n_pairs_per=1, device='cpu'):
    """Sample overlapping seqpairs"""

    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    h_LCH_masked = batch['historical_seq_LCH_masked'].to(device)
    f_emb = batch['future_seq_emb'].to(device)

    # Reconstructed LCH from featurizer forward pass
    with torch.no_grad():
        h_recon, *_ = featurizer(h_emb, h_LCH_masked, f_emb)

    seq_len = h_emb.shape[1]
    
    buffer = np.ceil(sample_len*1.5).astype(int)
    
    # Randomly sample centers for LHS and RHS, so the windows overlap by at least 16 points
    centers_a = np.random.randint(low=buffer, high=seq_len - buffer, size=n_pairs_per)
    centers_b = centers_a + np.random.randint(low=1, high=sample_len + 1, size=n_pairs_per) - sample_len//2
    
    # Slice to LHS windows
    lhs_emb, lhs_LCH = [], []
    for c in centers_a:
        lhs_emb.append(h_emb[:, c - sample_len//2:c + sample_len//2, :])
        lhs_LCH.append(h_LCH[:, c - sample_len//2:c + sample_len//2, :])
    lhs_emb = torch.cat(lhs_emb, dim=0)
    lhs_LCH = torch.cat(lhs_LCH, dim=0)

    # Slide to RHS windows  
    rhs_emb, rhs_LCH = [], []
    for c in centers_b:
        rhs_emb.append(h_emb[:, c - sample_len//2:c + sample_len//2, :])
        rhs_LCH.append(h_recon[:, c - sample_len//2:c + sample_len//2, :])
    rhs_emb = torch.cat(rhs_emb, dim=0)
    rhs_LCH = torch.cat(rhs_LCH, dim=0)

    # Add Gaussian noise
    lhs_means, lhs_stds = torch.mean(lhs_LCH, dim=1, keepdim=True), torch.std(lhs_LCH, dim=1, keepdim=True)
    rhs_means, rhs_stds = torch.mean(rhs_LCH, dim=1, keepdim=True), torch.std(rhs_LCH, dim=1, keepdim=True)
    means, stds = (lhs_means + rhs_means)/2, (lhs_stds + rhs_stds)/2
    lhs_LCH = lhs_LCH + torch.randn_like(lhs_LCH)*stds + means
    rhs_LCH = rhs_LCH + torch.randn_like(rhs_LCH)*stds + means
    
    return (lhs_emb, lhs_LCH), (rhs_emb, rhs_LCH)


def constrastive_loss(x, y, temp=1):

    # Temperature scaled similarity scores
    sims_xy = torch.exp(torch.matmul(x, y.T)/temp)
    sims_xx = torch.exp(torch.matmul(x, x.T)/temp)
    sims_yy = torch.exp(torch.matmul(y, y.T)/temp)

    # Positive pair markers
    pp_markers = torch.eye(len(x), device=x.device)

    # Positive pair scores
    pp_scores = torch.sum(sims_xy*pp_markers, dim=-1)

    # Negative samples scores
    ns_scores = (
        torch.sum(sims_xy*(1 - pp_markers), dim=-1) +
        torch.sum(sims_xx*(1 - pp_markers), dim=-1) +
        torch.sum(sims_yy*(1 - pp_markers), dim=-1)
    )

    # Positive score to negative samples score ratios
    ratios = pp_scores/ns_scores

    # normalized temperature-scaled cross entropy loss
    loss = -torch.log(torch.sum(ratios))

    return loss


def feature_learning_loss(model, batch, device='cpu'):
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    h_LCH_masked = batch['historical_seq_LCH_masked'].to(device)
    recon_loss_mask = batch['recon_loss_mask'].to(device)
    
    f_emb = batch['future_seq_emb'].to(device)
    f_LCH = batch['future_seq_LCH'].to(device)
    f_ret_dists = batch['future_ret_dists'].to(device)

    # Forward
    yh_h_LCH, yh_f_LCH, yh_f_ret_probas = model(h_emb, h_LCH_masked, f_emb)

    # Reconstruction loss
    loss_recon = torch.abs(yh_h_LCH - h_LCH)
    loss_recon = torch.sum(loss_recon*recon_loss_mask)/torch.sum(recon_loss_mask)/3

    # Future return path and distribution losses
    loss_LCH = torch.mean(torch.mean(torch.abs(yh_f_LCH - f_LCH), dim=(1,2)))
    loss_dists = nn.BCELoss()(yh_f_ret_probas, f_ret_dists)

    # Scale and combine losses
    loss = (
        loss_recon*0.25 + 
        loss_LCH*1.5 + 
        loss_dists
    )

    losses = (
        loss_recon.item(), 
        loss_LCH.item(),
        loss_dists.item(),
        loss
    )

    return losses


def train_feat_batch(model, optimizer, loss_fxn, batch, device='cpu'):

    *component_losses, loss = loss_fxn(model, batch, device=device)

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()
    
    losses = (*component_losses, loss.item())
    return losses


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


def ret_distribution_loss(LCH_samples, ret_probas):
    """Measures the mean squared distances between the qth sample 
    and the target value for that quantile for each period in the 
    generated sample of sequences.
    
    Quantiles are given by the predicted return distribution probabilities
    
    Args:
        LCH_samples (tensor): Generated samples (batch_size, n_samples, seq_len, 3)
        ret_probas (tensor): Predicted cumulative return distribution (batch_size, seq_len, n_dist_targets)
    """
    
    batch_size, seq_len, n_dist_targets = ret_probas.shape
    
    # Target values to compare qth sample values to
    targets = torch.tensor(
        norm.ppf(np.linspace(0.001, 0.999, n_dist_targets)),
        dtype=torch.float32,
        device=LCH_samples.device
    )
    
    # Cumulative close-to-close norms of samples
    CC_rets = LCH_samples[:,:,:,1]
    cumulative_CC_rets = torch.cumsum(CC_rets, dim=-1)
    vol_normalizers = (torch.arange(seq_len) + 1)**0.5  # Volatility scales with the sqrt of time
    cumulative_CC_norms = cumulative_CC_rets/vol_normalizers
    
    losses = []
    # Loop over each item in batch
    for i in range(batch_size):
        # Loop over each period in sequence
        for j in range(seq_len):
            samples_ij = cumulative_CC_norms[i,:,j]
            q_ij = ret_probas[i,j,:]  # Predicted quantiles ith sample, jth period
            quantiles_ij = torch.quantile(samples_ij, q=q_ij)  # Quantile values from generated samples
            losses.append(torch.mean((quantiles_ij - targets)**2))  # Spread between generated values and targets

    return sum(losses)/len(losses)


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
    n_critic_repeats=1,
    n_gen_repeats=1,
    n_samples=(4,500),
    device='cpu'):

    featurizer, generator, critic = models
    gen_opt, critic_opt = optimizers

    # Batch data
    h_emb = batch['historical_seq_emb'].to(device)
    h_LCH = batch['historical_seq_LCH'].to(device)
    f_emb = batch['future_seq_emb'].to(device)
    LCH_real = batch['future_seq_LCH'].to(device)
    

    ### Get context ###
    with torch.no_grad():
        context = featurizer.encode_future_path(h_emb, h_LCH, f_emb)
        ret_probas = featurizer.return_distribution_probas(context)  


    ### Update critic ###
    critic_losses_real, critic_losses_fake = [], []
    for _ in range(n_critic_repeats):

        # Generate fake examples
        with torch.no_grad():
            noise = generator.gen_noise(*f_emb.shape[:2], device=device)
            LCH_fake = generator(context, f_emb, noise)
        
        # Calculate scores
        scores_real = critic(context, f_emb, LCH_real)
        scores_fake = critic(context, f_emb, LCH_fake)

        # Critic loss
        loss_real = torch.mean(torch.mean((scores_real - 1)**2, dim=-1))
        loss_fake = torch.mean(torch.mean((scores_fake - 0)**2, dim=-1))
        critic_losses_real.append(loss_real.item())
        critic_losses_fake.append(loss_fake.item())
        real_wt = np.exp(np.log(loss_real.item())*10)
        fake_wt = np.exp(np.log(loss_fake.item())*10)
        real_wt = real_wt/(real_wt + fake_wt)
        fake_wt = 1 - real_wt
        critic_loss = loss_real*real_wt + loss_fake*fake_wt
        
        # Backprop
        critic_loss.backward()
        
        # Update critic's weights
        critic_opt.step()
        critic_opt.zero_grad()

    
    ### Update generator ###
    gen_losses, dist_losses = [], []
    for _ in range(n_gen_repeats):

        noise = generator.gen_noise(*f_emb.shape[:2], device=device)
        LCH_fake = generator(context, f_emb, noise)
        scores_fake = critic(context, f_emb, LCH_fake)
        gen_loss = torch.mean(torch.mean((scores_fake - 1)**2, dim=-1))
        gen_losses.append(gen_loss.item())

        # Distribution loss
        sample_idxs = torch.randint(low=0, high=context.shape[0], size=n_samples[:1])
        with torch.no_grad():
            ret_probas = featurizer.return_distribution_probas(context[sample_idxs])  
        LCH_samples = generator.generate_samples(context[sample_idxs], f_emb[sample_idxs], n_samples[-1])
        dist_loss = ret_distribution_loss(LCH_samples, ret_probas)
        dist_losses.append(dist_loss.item())
        
        # Total generator loss
        total_gen_loss = gen_loss + dist_loss
    
        # Backprop
        total_gen_loss.backward()

        # Update generator's weights
        gen_opt.step()
        gen_opt.zero_grad()

    return (
        np.mean(critic_losses_real)**0.5, 
        np.mean(critic_losses_fake)**0.5, 
        np.mean(gen_losses)**0.5,
        np.mean(dist_losses)
        
    )