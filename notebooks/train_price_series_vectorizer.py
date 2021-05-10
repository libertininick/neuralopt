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

from data_utils import PriceSeriesDataset
from modules import PriceSeriesFeaturizer, PriceSeriesVectorizer
from training_utils import sample_contrastive_pairs, constrastive_loss, train_feat_batch, lr_schedule, set_lr
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
with open(f'../models/featurizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)

historical_seq_len = model_config.get('historical_seq_len', 512)
future_seq_len = model_config.get('future_seq_len', 32)
n_dist_targets = model_config.get('n_dist_targets', 51)

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

# # Load featurizer

with open(f'../models/featurizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)
featurizer = PriceSeriesFeaturizer(**model_config)
featurizer.load_state_dict(torch.load('../models/featurizer/wts.pth'))
featurizer.eval()

# # Train

# +
# Set device
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

model_config = {
    'in_features': featurizer.n_features,
    'seq_len': future_seq_len,
    'out_features': int(future_seq_len**0.5),
    'dropout': 0,
}
with open('../models/vectorizer/model_config.json', 'w') as fp:
    json.dump(model_config, fp)
    
model = PriceSeriesVectorizer(encoding_fxn=featurizer.encode, **model_config)
optimizer = torch.optim.Adam(model.parameters())
model.to(device)
# -

# ## Testing

# ### Contrastive loss

# +
batch = next(iter(DataLoader(datasets['train'], batch_size=128, shuffle=True)))
lhs, rhs = sample_contrastive_pairs(batch, featurizer, future_seq_len, 8, device)


lhs_v = model(*lhs)
rhs_v = model(*rhs)
constrastive_loss(lhs_v, rhs_v, temp=0.1)
# -

# ### Vectorization manifold and cosine sim

batch = next(iter(DataLoader(datasets['train'], batch_size=1000, shuffle=True)))
with torch.no_grad():
    v = model(batch['future_seq_emb'], batch['future_seq_LCH']).numpy()
v.shape

from eval_utils import Manifold
manifold = Manifold(k=5).fit(v)
fig, ax = manifold.plot_manifold(figsize=(15,15))

# +
from scipy.spatial.distance import cdist

idx=3
dists = cdist(v, v[idx][None,:], metric='cosine').squeeze()

# Similarity hist
fig, ax = plt.subplots(figsize=(15,5))
_ = ax.hist(
    1 - dists[np.arange(len(dists)) != idx], 
    bins=30, 
    edgecolor='black',
    alpha=0.5
)
_ = ax.set_xlim(-1,1)

idxs = np.argsort(dists)
fig, ax = plt.subplots(figsize=(15,7))
_ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idx,:,1]).tolist(), color='red')
for i in range(1,4):
    _ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idxs[i],:,1]).tolist(), color='black', alpha=0.25)
# -

# ## Training Loop

# +
n_pairs_per = int(historical_seq_len/future_seq_len/2)
batch_size = 128
cycle_len = 10000//batch_size
lrs = lr_schedule(
    n_steps=len(datasets['train'])//batch_size + 1, 
    lr_min=0.00001, 
    lr_max=0.003
)
loss_quantiles = [0.05,0.25,0.5,0.75,0.95]

st = time.time()
best_train_loss, best_valid_loss = np.inf, np.inf
for e in range(3):
    # Training Loop
    model.train()
    train_losses = []
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):
        
        set_lr(optimizer, lrs[b_i])
        
        # Sample contrastive pairs
        lhs, rhs = sample_contrastive_pairs(batch, featurizer, future_seq_len, n_pairs_per, device)
        
        # Vectorize
        lhs_v = model(*lhs)
        rhs_v = model(*rhs)
        
        # Contrastive loss
        loss_i = constrastive_loss(lhs_v, rhs_v, temp=0.1)
        train_losses.append(loss_i.item())
        
        # Backprop
        loss_i.backward()
        optimizer.step()
        
        # Cleanup
        optimizer.zero_grad()
    
        if (b_i + 1) % cycle_len == 0:
            print(
                e,
                f'{b_i + 1:>10,}',
                f'{(time.time() - st)/60:>7.2f}m',
                np.quantile(train_losses[-cycle_len:], q=loss_quantiles).round(3),
            )

    # Validation Loop
    model.eval()
    valid_losses = []
    loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):
        # Sample contrastive pairs
        lhs, rhs = sample_contrastive_pairs(batch, featurizer, future_seq_len, n_pairs_per, device)
        
        with torch.no_grad():
            lhs_v = model(*lhs)
            rhs_v = model(*rhs)
            loss_i =  constrastive_loss(lhs_v, rhs_v, temp=0.1)
            valid_losses.append(loss_i.item())

    print(
        e,
        f'Validation',
        f'{(time.time() - st)/60:>7.2f}m',
        np.quantile(valid_losses, q=loss_quantiles).round(3),
    )
    
    train_loss, valid_loss = np.mean(train_losses), np.mean(valid_losses)
    if train_loss <= best_train_loss and valid_loss <= best_valid_loss:
        torch.save(model.state_dict(), '../models/vectorizer/wts.pth')
        best_train_loss, best_valid_loss = train_loss, valid_loss
        print('*** Model saved')
    else:
        print('!!! Model NOT saved')


# -
# # Eval

# +
with open('../models/vectorizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)
    
model = PriceSeriesVectorizer(encoding_fxn=featurizer.encode, **model_config)
model.load_state_dict(torch.load('../models/vectorizer/wts.pth'))
model.eval()

batch = next(iter(DataLoader(datasets['train'], batch_size=5000, shuffle=True)))
with torch.no_grad():
    v = model(batch['future_seq_emb'], batch['future_seq_LCH']).numpy()
v.shape
# -

# ## Manifold

manifold = Manifold(k=5).fit(v)
fig, ax = manifold.plot_manifold(figsize=(15,15))

# ## Cosine sim

# +
idx=12
dists = cdist(v, v[idx][None,:], metric='cosine').squeeze()

# Similarity hist
fig, ax = plt.subplots(figsize=(15,5))
_ = ax.hist(
    1 - dists[np.arange(len(dists)) != idx], 
    bins=30, 
    edgecolor='black',
    alpha=0.5
)
_ = ax.set_xlim(-1,1)

idxs = np.argsort(dists)
fig, ax = plt.subplots(figsize=(15,7))
_ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idx,:,1]).tolist(), color='red')
for i in range(1,6):
    print(dists[idxs[i]])
    _ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idxs[i],:,1]).tolist(), color='black', alpha=0.25)
# -


