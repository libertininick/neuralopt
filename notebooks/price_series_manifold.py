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
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import umap

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
from eval_utils import SequenceVectorizer
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
historical_seq_len = 512
future_seq_len = 32
n_dist_targets = 27

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

# # Fit Sequence Vectorizer

# ## Load Price Series Featurizer

# +
featurizer = PriceSeriesFeaturizer(
    n_features=64,
    historical_seq_len=historical_seq_len,
    future_seq_len=future_seq_len,
    n_dist_targets=n_dist_targets,
    conv_kernel_size=5,
    n_attention_heads=2,
    rnn_kernel_size=32,
    n_blocks=3
)

featurizer.load_state_dict(torch.load('../models/price_series_featurizer_wts.pth'))
featurizer.eval()
# -

# ## Sample inputs for fitting Vectorizer

batch = next(iter(DataLoader(datasets['train'], batch_size=1000, shuffle=True)))

# ## Fit

sv = SequenceVectorizer(
    enc_func=featurizer.encode, 
    n_common=2, 
    n_ref_samples=500, 
    seed=1234
).fit(batch['historical_seq_emb'], batch['historical_seq_LCH'])

# ## Transform

v = sv.vectorize_seq(batch['future_seq_emb'], batch['future_seq_LCH'])

# ## 2-D plot

# +
reducer = umap.UMAP(
    n_components=2,
    random_state=1234
)

v_2d = reducer.fit_transform(v)
# -

fig, ax = plt.subplots(figsize=(12,12))
_ = ax.scatter(*v_2d.T, alpha=0.5)

# ## K-nearest neighbor sequences

# +
idx=7
idxs = np.argsort(cdist(v, v[idx][None,:]).squeeze())

fig, ax = plt.subplots(figsize=(15,7))

_ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idx,:,1]).tolist(), color='red')

for i in range(1,6):
    _ = ax.plot([0]+np.cumsum(batch['future_seq_LCH'][idxs[i],:,1]).tolist(), color='black', alpha=0.25)
# -

# # Estimate manifold

k = 5
r = v[:500]
g = v[500:]

manifold_r = Manifold(k).fit(r)
fig, ax = manifold_r.plot_manifold()

manifold_g = Manifold(k).fit(g)
fig, ax = manifold_g.plot_manifold()

# +
# Precision
precision = manifold_r.predict(g)
print(f'{np.mean(precision): .2%}')

# Recall
recall = manifold_g.predict(r)
print(f'{np.mean(recall): .2%}')
# -


