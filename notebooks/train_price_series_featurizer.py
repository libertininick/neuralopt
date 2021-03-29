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
from training_utils import calculate_loss, train_batch, lr_schedule, set_lr
# -

# # Data Set

# +
p = re.compile(r'[^\\\\|\/]{1,100}(?=\.pkl$)')

files = np.array(glob.glob('D:/opt/Price Transforms/*.pkl'))
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
historical_seq_len = 2*260
future_seq_len = 260//4
n_dist_targets = 13

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
# -

# # Baseline losses

# +
baseline_recon_losses = []
baseline_LCH_losses = []
baseline_mag_losses = []
baseline_dists_losses = []
for sym_i, (symbol, windows) in enumerate(datasets['train'].symbol_windows.items(),1):
    if len(windows) > 10:
        _recon_losses = []
        _seq_LCH = []
        _ret_dists = []
        
        batch_size = len(windows)
        for i in range(batch_size):
            _item = datasets['train'].__getitem__(i, symbol)
            
            _recon_prediction = torch.mean(_item['historical_seq_LCHVT'][:,:3], dim=0, keepdim=True)
            _recon_loss = torch.abs(_item['historical_seq_LCHVT'][:,:3] - _recon_prediction)
            _recon_mask = _item['recon_loss_mask']
            _recon_losses.append((torch.sum(_recon_loss*_recon_mask)/torch.sum(_recon_mask)/3).item())
            
            _seq_LCH.append(_item['future_seq_LCH'])
            _ret_dists.append(_item['future_ret_dists'])
            
        baseline_recon_losses.append(np.array(_recon_losses))

        _seq_LCH = torch.stack(_seq_LCH, dim=0)
        _mag = torch.abs(_seq_LCH)
        _ret_dists = torch.stack(_ret_dists, dim=0)
        
        _LCH_losses, _mag_losses, _dists_losses = [], [], []
        for i in range(batch_size):
            baseline_mask = torch.arange(batch_size) != i
            
            _seq_LCH_baseline = torch.mean(_seq_LCH[baseline_mask], dim=0)
            _LCH_losses.append(torch.mean(torch.abs(_seq_LCH[i] - _seq_LCH_baseline)).item())
            
            _mag_baseline = torch.mean(torch.abs(_seq_LCH[baseline_mask]), dim=0)
            _mag_losses.append(torch.mean(torch.abs(torch.abs(_seq_LCH[i]) - _mag_baseline)).item())
            
            _ret_dists_baseline = torch.mean(_ret_dists[baseline_mask], dim=0)
            _dists_losses.append(nn.BCELoss()(_ret_dists_baseline, _ret_dists[i]).item())
        
        baseline_LCH_losses.append(np.array(_LCH_losses))
        baseline_mag_losses.append(np.array(_mag_losses))
        baseline_dists_losses.append(np.array(_dists_losses))
      
    if sym_i % (len(datasets['train'].symbol_windows)//20) == 0:
        print(f'''{sym_i/len(datasets['train'].symbol_windows):.0%}''', end=' ')
    
baseline_recon_losses = np.concatenate(baseline_recon_losses).astype(float)
baseline_LCH_losses = np.concatenate(baseline_LCH_losses).astype(float)
baseline_mag_losses = np.concatenate(baseline_mag_losses).astype(float)
baseline_dists_losses = np.concatenate(baseline_dists_losses).astype(float)

# +
rnd = np.random.RandomState(1234)
batch_size = 16
n_batches = len(baseline_dists_losses)//batch_size

baseline_recon_losses = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_recon_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_LCH_losses = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_LCH_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_mag_losses = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_mag_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

baseline_dists_losses = np.mean(
    np.stack(
        np.split(
            rnd.choice(baseline_dists_losses, n_batches*batch_size, replace=False), 
            n_batches
        ), 
        axis=0
    ), axis=-1
)

print('Recon:', np.quantile(baseline_recon_losses, q=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
print('LCH: ', np.quantile(baseline_LCH_losses, q=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
print('Mag:  ', np.quantile(baseline_mag_losses, q=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
print('Dists:', np.quantile(baseline_dists_losses, q=[0.05, 0.25, 0.5, 0.75, 0.95]).round(3))
# -

baseline_losses = {
    'recon': np.quantile(baseline_recon_losses, q=[0.25, 0.5, 0.75,]),
    'LCH': np.quantile(baseline_LCH_losses, q=[0.25, 0.5, 0.75,]),
    'mag': np.quantile(baseline_mag_losses, q=[0.25, 0.5, 0.75,]),
    'dists': np.quantile(baseline_dists_losses, q=[0.25, 0.5, 0.75,]),
}
baseline_losses

# # Train

model = PriceSeriesFeaturizer(
    n_features=64,
    historical_seq_len=historical_seq_len,
    future_seq_len=future_seq_len,
    n_dist_targets=n_dist_targets,
    conv_kernel_size=3,
    n_attention_heads=1,
    rnn_kernel_size=20,
    n_blocks=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# ## Testing

batch = next(iter(DataLoader(datasets['train'], batch_size=16, shuffle=True)))
calculate_loss(model, batch)

# ## Training Loop

# +
batch_size = 16
cycle_len = 500

losses = {
    'recon': [],
    'LCH': [],
    'mag': [],
    'dists': [],
    'total': [],
}

model.train()

st = time.time()
for e in range(5):
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True)
    for i, batch in enumerate(loader):
        losses_i = train_batch(model, optimizer, batch)
        for ll, l_i in zip(losses.values(), losses_i):
            ll.append(l_i)

        if (i + 1) % cycle_len == 0:
            print(
                e,
                f'{i + 1:<7}',
                f'{(time.time() - st)/60:>7.2f}m',
                (np.quantile(losses['recon'][-cycle_len:], q=[0.25,0.5,0.75])/baseline_losses['recon']).round(3),
                (np.quantile(losses['LCH'][-cycle_len:], q=[0.25,0.5,0.75])/baseline_losses['LCH']).round(3),
                (np.quantile(losses['mag'][-cycle_len:], q=[0.25,0.5,0.75])/baseline_losses['mag']).round(3),
                (np.quantile(losses['dists'][-cycle_len:], q=[0.25,0.5,0.75])/baseline_losses['dists']).round(3),
            )


# -
torch.save(model.state_dict(), 'price_series_featurizer_wts.pth')

# # Eval

# ## Validation losses

# +
batch_size = 16

valid_losses = {
    'recon': [],
    'LCH': [],
    'mag': [],
    'dists': [],
}

model.eval()

st = time.time()
for e in range(5):
    for i, batch in enumerate(DataLoader(datasets['valid'], batch_size=batch_size, shuffle=False)):
        
        with torch.no_grad():
            losses_i = calculate_loss(model, batch)
            
        for ll, l_i in zip(valid_losses.values(), losses_i):
            ll.append(l_i)
            
print(
    (np.quantile(valid_losses['recon'], q=[0.25,0.5,0.75])/baseline_losses['recon']).round(3),
    (np.quantile(valid_losses['LCH'], q=[0.25,0.5,0.75])/baseline_losses['LCH']).round(3),
    (np.quantile(valid_losses['mag'], q=[0.25,0.5,0.75])/baseline_losses['mag']).round(3),
    (np.quantile(valid_losses['dists'], q=[0.25,0.5,0.75])/baseline_losses['dists']).round(3),
)
# -

# ## Historical weights viewer

# +
with torch.no_grad():
    wts = model._avg_wts().numpy().T
    
fig, ax = plt.subplots(figsize=(7,15))
_  = ax.imshow(wts)
# -

# ## Eval training batch

# +
batch = next(iter(DataLoader(datasets['train'], batch_size=128, shuffle=True)))

baseline_LCH = torch.mean(batch['future_seq_LCH'][:,:,1], dim=0).numpy()
baseline_path = np.append([0], np.cumsum(baseline_LCH, axis=-1))

with torch.no_grad():
    x_recon = model.encode_decode(batch['historical_seq_emb'], batch['historical_seq_LCHVT_masked'])
    f_LCH, f_ret_probas = model.forecast(
        batch['historical_seq_emb'], 
        batch['historical_seq_LCHVT'],  
        batch['future_seq_emb']
    )
    
# -


# ## Historical recon viewer

fig, axs = plt.subplots(ncols=3, figsize=(5,15))
_ = axs[0].imshow(batch['historical_seq_LCHVT'][0, :40, :3], cmap='RdBu', vmin=-3, vmax=3)
_ = axs[1].imshow(batch['historical_seq_LCHVT_masked'][0, :40, :3], cmap='RdBu', vmin=-3, vmax=3)
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

# ### Day 65

fig, ax = plt.subplots(figsize=(15,7))
for i in range(len(f_ret_probas)):
    _ = ax.plot(f_ret_probas[i,-1,:].numpy(), '-o', color='black', alpha=0.05)

# # Generation

from modules import PriceSeriesGenerator

z_dim = 512
g = PriceSeriesGenerator(z_dim, n_future, 64, 3, 1)

batch_size = 4
batch = next(iter(DataLoader(datasets['train'], batch_size=4, shuffle=True)))
noise = torch.randn((batch_size, z_dim))

batch['future_LCH']

with torch.no_grad():
    context =  model.encode_future_path(batch['historical_seq_emb'], batch['historical_seq'],  batch['future_seq_emb'])

context.shape

with torch.no_grad():
    samples = g(context, noise)

samples.shape

samples[0]


