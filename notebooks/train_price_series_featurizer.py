# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python [conda env:neuralopt_env] *
#     language: python
#     name: conda-env-neuralopt_env-py
# ---

# # Imports

# +
# %load_ext autoreload
# %autoreload 2

import glob
import json
import os
import re
import sys
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)
p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

from data_utils import get_data_splits, PriceSeriesDataset
from modules import PriceSeriesFeaturizer
from training_utils import feature_learning_loss, train_feat_batch, lr_schedule, set_lr
# -

# # Data Set

# +
p = re.compile(r'[^\\\\|\/]{1,100}(?=\.pkl$)')

files = np.array(glob.glob('D:/opt/price_transforms/*.pkl'))
symbols = np.array([p.findall(file)[0] for file in files])

rnd = np.random.RandomState(1234)

n = len(files)
n_valid = 100
n_test = 100
n_train = n - n_valid - n_test

idxs_all = np.arange(n)
idxs_train = rnd.choice(idxs_all, n_train, replace=False)
idxs_other = np.setdiff1d(idxs_all, idxs_train)
idxs_valid = rnd.choice(idxs_other, n_valid, replace=False)
idxs_test = np.setdiff1d(idxs_other, idxs_valid)

# +
historical_seq_len = 512
future_seq_len = 32
n_dist_targets = 51

datasets = {
    'train': PriceSeriesDataset(
        symbols=symbols[idxs_train],
        files=files[idxs_train],
        n_historical=historical_seq_len,
        n_future=future_seq_len,
        n_dist_targets=n_dist_targets,
    ),
    'valid': PriceSeriesDataset(
        symbols=symbols[idxs_valid],
        files=files[idxs_valid],
        n_historical=historical_seq_len,
        n_future=future_seq_len,
        n_dist_targets=n_dist_targets,
    ),
    'test': PriceSeriesDataset(
        symbols=symbols[idxs_test],
        files=files[idxs_test],
        n_historical=historical_seq_len,
        n_future=future_seq_len,
        n_dist_targets=n_dist_targets,
    ),
}

print(f'''Train: {len(datasets['train']):>12,}''')
print(f'''Valid: {len(datasets['valid']):>12,}''')
print(f'''Test : {len(datasets['test']):>12,}''')

# + [markdown] heading_collapsed=true
# # Baseline losses

# + hidden=true
baseline_recon_losses = []
baseline_LCH_losses = []
baseline_dists_losses = []
for sym_i, (symbol, windows) in enumerate(datasets['train'].symbol_windows.items(),1):
    if len(windows) > 10:
        _recon_losses = []
        _seq_LCH = []
        _ret_dists = []
        
        batch_size = len(windows)
        for i in range(batch_size):
            _item = datasets['train'].__getitem__(i, symbol)
            
            _recon_prediction = torch.mean(_item['historical_seq_LCH'], dim=0, keepdim=True)
            _recon_loss = torch.abs(_item['historical_seq_LCH'] - _recon_prediction)
            _recon_mask = _item['recon_loss_mask']
            _recon_losses.append((torch.sum(_recon_loss*_recon_mask)/torch.sum(_recon_mask)/3).item())
            
            _seq_LCH.append(_item['future_seq_LCH'])
            _ret_dists.append(_item['future_ret_dists'])
            
        baseline_recon_losses.append(np.array(_recon_losses))

        _seq_LCH = torch.stack(_seq_LCH, dim=0)
        _ret_dists = torch.stack(_ret_dists, dim=0)
        
        _LCH_losses, _mag_losses, _dists_losses = [], [], []
        for i in range(batch_size):
            baseline_mask = torch.arange(batch_size) != i
            
            _seq_LCH_baseline = torch.mean(_seq_LCH[baseline_mask], dim=0)
            _LCH_losses.append(torch.mean(torch.abs(_seq_LCH[i] - _seq_LCH_baseline)).item())
            
            _ret_dists_baseline = torch.mean(_ret_dists[baseline_mask], dim=0)
            _dists_losses.append(nn.BCELoss()(_ret_dists_baseline, _ret_dists[i]).item())
        
        baseline_LCH_losses.append(np.array(_LCH_losses))
        baseline_dists_losses.append(np.array(_dists_losses))
      
    if sym_i % (len(datasets['train'].symbol_windows)//20) == 0:
        print(f'''{sym_i/len(datasets['train'].symbol_windows):.0%}''', end=' ')
    
baseline_recon_losses = np.concatenate(baseline_recon_losses).astype(float)
baseline_LCH_losses = np.concatenate(baseline_LCH_losses).astype(float)
baseline_dists_losses = np.concatenate(baseline_dists_losses).astype(float)

# + hidden=true
rnd = np.random.RandomState(1234)
batch_size = 16
n_batches = len(baseline_dists_losses)//batch_size

baseline_recon_losses_b = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_recon_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_LCH_losses_b = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_LCH_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_dists_losses_b = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_dists_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_losses = {
    'recon': np.quantile(baseline_recon_losses_b, q=[0.25, 0.5, 0.75,]),
    'LCH': np.quantile(baseline_LCH_losses_b, q=[0.25, 0.5, 0.75,]),
    'dists': np.quantile(baseline_dists_losses_b, q=[0.25, 0.5, 0.75,]),
}
baseline_losses
# -

# # Train

# +
model_config = {
    'n_features': 128,
    'historical_seq_len': 512,
    'future_seq_len': 32,
    'n_dist_targets': 51,
    'dropout': 0
}

with open(f'../models/featurizer/model_config.json', 'w') as fp:
    json.dump(model_config, fp)

model = PriceSeriesFeaturizer(**model_config)
optimizer = torch.optim.Adam(model.parameters())
# -

# ## Testing

batch = next(iter(DataLoader(datasets['train'], batch_size=128, shuffle=True)))
feature_learning_loss(model, batch)

# + [markdown] heading_collapsed=true
# ## Training Loop

# + hidden=true
batch_size = 128
cycle_len = 100
lrs = lr_schedule(
    n_steps=len(datasets['train'])//batch_size + 1, 
    lr_min=0.00001, 
    lr_max=0.003
)
q = [0.05,0.25,0.5,0.75,0.95]

st = time.time()
best_train_loss, best_valid_loss = np.inf, np.inf
for e in range(20):
    # Training Loop
    model.train()
    train_losses = {'recon': [], 'LCH': [], 'dists': [], 'total': []}
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):
        
        set_lr(optimizer, lrs[b_i])
        
        losses_i = train_feat_batch(model, optimizer, feature_learning_loss, batch)
        for ll, l_i in zip(train_losses.values(), losses_i):
            ll.append(l_i)

        if (b_i + 1) % cycle_len == 0:
            print(
                e,
                f'{b_i + 1:>10,}',
                f'{(time.time() - st)/60:>7.2f}m',
                np.quantile(train_losses['recon'][-cycle_len:], q=q).round(3),
                np.quantile(train_losses['LCH'][-cycle_len:], q=q).round(3),
                np.quantile(train_losses['dists'][-cycle_len:], q=q).round(3),
            )

    # Validation Loop
    model.eval()
    valid_losses = {'recon': [], 'LCH': [], 'dists': [], 'total': []}
    loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):
        with torch.no_grad():
            *component_losses, total_loss = feature_learning_loss(model, batch)
        losses_i = (*component_losses, total_loss.item())
        for ll, l_i in zip(valid_losses.values(), losses_i):
            ll.append(l_i)
    print(
        e,
        f'Validation',
        f'{(time.time() - st)/60:>7.2f}m',
        np.quantile(valid_losses['recon'], q=q).round(3),
        np.quantile(valid_losses['LCH'], q=q).round(3),
        np.quantile(valid_losses['dists'], q=q).round(3),
    )
    
    train_loss, valid_loss = np.mean(train_losses['total']), np.mean(valid_losses['total'])
    if train_loss <= best_train_loss and valid_loss <= best_valid_loss:
        torch.save(model.state_dict(), '../models/featurizer/wts_2.pth')
        best_train_loss, best_valid_loss = train_loss, valid_loss
        print('*** Model saved')
    else:
        print('!!! Model NOT saved')


# -
# # Eval

with open(f'../models/featurizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)
model = PriceSeriesFeaturizer(**model_config)
model.load_state_dict(torch.load('../models/featurizer/wts.pth'))
model.eval()

# ## Test losses

# +
test_losses = {'recon': [], 'LCH': [], 'dists': [], 'total': []}
for b_i, batch in enumerate(DataLoader(datasets['test'], batch_size=128, shuffle=True, num_workers=3)):
    with torch.no_grad():
        *component_losses, total_loss = feature_learning_loss(model, batch)
    losses_i = (*component_losses, total_loss.item())
    for ll, l_i in zip(test_losses.values(), losses_i):
        ll.append(l_i)

q = [0.05,0.25,0.5,0.75,0.95]
print(
    np.quantile(test_losses['recon'], q=q).round(3),
    np.quantile(test_losses['LCH'], q=q).round(3),
    np.quantile(test_losses['dists'], q=q).round(3),
)
# -

# ## Historical weights viewer

# +
with torch.no_grad():
    wts = model._avg_wts().numpy().T

fig, ax = plt.subplots(figsize=(15,10))
_ = ax.plot(
    [0] + list(np.cumsum(wts[:,0][::-1])),
    alpha=0.5,
    color='blue',
    label='Day 1'
)
_ = ax.plot(
    [0] + list(np.cumsum(wts[:,1][::-1])),
    alpha=0.5,
    color='green',
    label='Day 2'
)
_ = ax.plot(
    [0] + list(np.cumsum(wts[:,5][::-1])),
    alpha=0.5,
    color='orange',
    label='Day 5'
)
_ = ax.plot(
    [0] + list(np.cumsum(wts[:,15][::-1])),
    alpha=0.5,
    color='red',
    label='Day 15'
)
_ = ax.plot(
    [0] + list(np.cumsum(wts[:,-1][::-1])),
    alpha=0.5,
    color='purple',
    label='Day 32'
)

_ = ax.plot(
    [0, 511],
    [0, 1],
    color='black',
    linestyle='--',
    label='Uniform',
)

_ = ax.set_xlabel('Historical Lag', fontsize=16)
_ = ax.legend()
# -

# ## Historical recon viewer

# +
batch = next(iter(DataLoader(datasets['train'], batch_size=128, shuffle=True)))

baseline_LCH = torch.mean(batch['future_seq_LCH'][:,:,1], dim=0).numpy()
baseline_path = np.append([0], np.cumsum(baseline_LCH, axis=-1))

with torch.no_grad():
    x_recon, f_LCH, f_ret_probas = model(
        batch['historical_seq_emb'], 
        batch['historical_seq_LCH_masked'],  
        batch['future_seq_emb']
    )
    
# -


fig, axs = plt.subplots(ncols=3, figsize=(5,15))
_ = axs[0].imshow(batch['historical_seq_LCH'][0, :40, :3], cmap='RdBu', vmin=-3, vmax=3)
_ = axs[1].imshow(batch['historical_seq_LCH_masked'][0, :40, :3], cmap='RdBu', vmin=-3, vmax=3)
_ = axs[2].imshow(x_recon[0][:40], cmap='RdBu', vmin=-3, vmax=3)

# ## Future path plotter

# +
fig, ax = plt.subplots(figsize=(15,10))
_ = ax.plot(baseline_path, '-o', color='red')

for i in range(len(f_LCH)):
    _ = ax.plot(np.append([0], np.cumsum(f_LCH[i,:,1].numpy())), '-', color='black', alpha=0.05)
    


# +
fig, ax = plt.subplots(figsize=(12,12))

_ = ax.imshow(np.corrcoef(f_LCH[:,:,1]), cmap='RdBu', vmin=-1, vmax=1)
# -


# ## Future dist plotter

# ### Day 1

fig, ax = plt.subplots(figsize=(15,7))
for i in range(len(f_ret_probas)):
    _ = ax.plot(f_ret_probas[i,0,:].numpy(), '-o', color='black', alpha=0.05)

# ### Day 10

fig, ax = plt.subplots(figsize=(15,7))
for i in range(len(f_ret_probas)):
    _ = ax.plot(f_ret_probas[i,9,:].numpy(), '-o', color='black', alpha=0.05)

# ### Day 32

fig, ax = plt.subplots(figsize=(15,7))
for i in range(len(f_ret_probas)):
    _ = ax.plot(f_ret_probas[i,-1,:].numpy(), '-o', color='black', alpha=0.05)


