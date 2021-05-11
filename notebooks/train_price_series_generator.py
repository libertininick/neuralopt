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
from modules import PriceSeriesFeaturizer, PriceSeriesVectorizer, PriceSeriesGenerator, PriceSeriesCritic
from training_utils import lr_schedule, set_lr, train_GC_least_square_batch
from eval_utils import Manifold, precision_recall_f1
# -

# # Data Sets

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

# # Load featurizer

with open(f'../models/featurizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)
featurizer = PriceSeriesFeaturizer(**model_config)
featurizer.load_state_dict(torch.load('../models/featurizer/wts.pth'))
featurizer.eval()

# # Load vectorizer

# +
with open('../models/vectorizer/model_config.json', 'r') as fp:
    model_config = json.load(fp)
    
vectorizer = PriceSeriesVectorizer(encoding_fxn=featurizer.encode, **model_config)
vectorizer.load_state_dict(torch.load('../models/vectorizer/wts.pth'))
vectorizer.eval()
# -

# # Vectorize "real" seq for precision/recall calculations

v_real = [] 
for batch in DataLoader(datasets['valid'], batch_size=128, shuffle=True, num_workers=3):
    with torch.no_grad():
        v = vectorizer(batch['future_seq_emb'], batch['future_seq_LCH'])
        v_real.append(v.numpy())
v_real = np.concatenate(v_real, axis=0)

# ## Precision/recall baseline

# +
k = 5
n = len(v_real)
idxs = np.arange(n)
rnd = np.random.RandomState(1234)

f1_scores = []
for _ in range(10):
    idxs_a = rnd.choice(idxs, size=n//2, replace=False)
    idxs_b = np.setdiff1d(idxs, idxs_a, assume_unique=True)
    
    v_a = v_real[idxs_a]
    v_b = v_real[idxs_b]
    
    *_, f1 = precision_recall_f1(v_a, v_b, k)
    f1_scores.append(f1)
    
f1_baseline = np.max(f1_scores)
f1_baseline
# -

# ## Validation manifold

manifold_real = Manifold(k=5).fit(v_real)
fig, ax = manifold_real.plot_manifold(figsize=(15,15))

# # Train

# +
noise_dim = 128
n_features=featurizer.n_features

generator = PriceSeriesGenerator(
    noise_dim=noise_dim, 
    n_features=n_features,
)

critic = PriceSeriesCritic(
    n_features=n_features,
)

gen_opt = torch.optim.Adam(generator.parameters())
critic_opt = torch.optim.Adam(critic.parameters())
# -

# ## Initial precision/recall

# +
v_fake = [] 
for batch in DataLoader(datasets['valid'], batch_size=128, shuffle=True, num_workers=3):
    
    noise = generator.gen_noise(*batch['future_seq_emb'].shape[:2])
    with torch.no_grad():
        context = featurizer.encode_future_path(
            batch['historical_seq_emb'],
            batch['historical_seq_LCH'],
            batch['future_seq_emb']
        )
        
        lch = generator(context, batch['future_seq_emb'], noise)
        v = vectorizer(batch['future_seq_emb'], lch)
        v_fake.append(v.numpy())

v_fake = np.concatenate(v_fake, axis=0)
# -

precision, recall, f1 = precision_recall_f1(v_real, v_fake, k)
print(f'Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}')

# ## Test batch

# +
loader = DataLoader(datasets['train'], shuffle=True, batch_size=128)
batch = next(iter(loader))

losses = train_GC_least_square_batch(
    (featurizer, generator, critic),
    (gen_opt, critic_opt), 
    batch,
)
losses
# -

# ## Training loop

# +
batch_size = 128
n_critic_repeats, n_gen_repeats = 3,1
cycle_len = 100

q=[0.25,0.5,0.75]
losses = {
    'critic_real': [],
    'critic_fake': [],
    'gen_real': [],
    'gen_dist': [],
}

lrs = lr_schedule(
    n_steps=len(datasets['train'])//batch_size + 1,
    lr_min=0.0001,
    lr_max=0.005
)
lr_mean = np.mean(lrs)

featurizer.eval()
critic.train()
st = time.time()
best_valid_f1 = 0
for e in range(5):
    
    # Training Loop
    generator.train()
    loader = DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):

        set_lr(critic_opt, lrs[b_i])
        set_lr(gen_opt, lrs[b_i])
        
        losses_i = train_GC_least_square_batch(
            (featurizer, generator, critic),
            (gen_opt, critic_opt), 
            batch, 
            n_critic_repeats=n_critic_repeats,
            n_gen_repeats=n_gen_repeats
        )
        for ll, l_i in zip(losses.values(), losses_i):
            ll.append(l_i)

        if (b_i + 1) % cycle_len == 0:
            # Swap repeats
            n_critic_repeats, n_gen_repeats = n_gen_repeats, n_critic_repeats
            
            print(
                e,
                f'{b_i + 1:<7}',
                f'{(time.time() - st)/60:>7.2f}m',
                np.quantile(losses['critic_real'][-cycle_len:], q=q).round(3),
                np.quantile(losses['critic_fake'][-cycle_len:], q=q).round(3),
                np.quantile(losses['gen_real'][-cycle_len:], q=q).round(3),
                np.quantile(losses['gen_dist'][-cycle_len:], q=q).round(3),
            )
            
    # Validation Loop
    generator.eval()
    v_fake = []
    loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=3)
    for b_i, batch in enumerate(loader):
        noise = generator.gen_noise(*batch['future_seq_emb'].shape[:2])
        
        with torch.no_grad():
            context = featurizer.encode_future_path(
                batch['historical_seq_emb'],
                batch['historical_seq_LCH'],
                batch['future_seq_emb']
            )

            lch = generator(context, batch['future_seq_emb'], noise)
            v = vectorizer(batch['future_seq_emb'], lch)
        v_fake.append(v.numpy())
    v_fake = np.concatenate(v_fake, axis=0)
    precision, recall, f1 = precision_recall_f1(v_real, v_fake, k)

    print(
        e,
        f'Validation',
        f'{(time.time() - st)/60:>7.2f}m',
        f'Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}'
    )
    
    if f1 > best_valid_f1:
        torch.save(generator.state_dict(), '../models/generator/wts.pth')
        best_valid_f1 = f1
        print('*** Model saved')
    else:
        print('!!! Model NOT saved')
# -

# # Eval

# ## Precision/recall

# +
generator.eval()

v_fake = [] 
for batch in DataLoader(datasets['valid'], batch_size=128, shuffle=True, num_workers=3):
    
    noise = generator.gen_noise(*batch['future_seq_emb'].shape[:2])
    with torch.no_grad():
        context = featurizer.encode_future_path(
            batch['historical_seq_emb'],
            batch['historical_seq_LCH'],
            batch['future_seq_emb']
        )
        
        lch = generator(context, batch['future_seq_emb'], noise)
    
    v_fake.append(vectorizer.vectorize_seq(batch['future_seq_emb'], lch))

v_fake = np.concatenate(v_fake, axis=0)
# -

manifold_fake = Manifold(k=5).fit(v_fake)
fig, ax = manifold_fake.plot_manifold(figsize=(15,15))

precision = manifold_real.predict(v_fake)
recall = manifold_fake.predict(v_real)
print(f'Precision: {np.mean(precision):.2%}, Recall: {np.mean(recall):.2%}')

# ## Single examples

# +
batch_size = 4
n_samples = 500

loader = DataLoader(datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=3)
batch = next(iter(loader))

with torch.no_grad():
    context = featurizer.encode_future_path(
        batch['historical_seq_emb'], 
        batch['historical_seq_LCH'], 
        batch['future_seq_emb']
    )
    LCH_samples = generator.generate_samples(context, batch['future_seq_emb'], n_samples=n_samples)

for i in range(batch_size):

    fig, ax = plt.subplots(figsize=(15,10))

    _ = ax.plot(np.append([0], np.cumsum(batch['future_seq_LCH'][i,:,1].numpy())), '-o', color='red')
    for j in range(n_samples):
        _ = ax.plot(np.append([0], np.cumsum(LCH_samples[i,j,:,1].numpy())), '-', color='black', alpha=0.05)


# -


