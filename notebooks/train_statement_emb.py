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

# +
# %load_ext autoreload
# %autoreload 2

import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
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

DATA_PATH = '../../../Investing Models/Data'

from data_utils import get_data_splits, FinancialStatementsDataset
# -

# # Data Splits
# 1-year windows, with 5-year buffer. No statement overlap for the 5-year history

splits = get_data_splits(
    data_dir='../../../Investing Models/Data/Financial Transforms', 
    st_date='1989-12-31', 
    end_date='2020-12-31', 
    window_len=365, 
    buffer=365*5, 
    p_valid=0.2
)

# # Data Sets

datasets = []
n_historical=20
p_mask=0.2
for i, split in enumerate(splits):
    datasets.append({
        'train': FinancialStatementsDataset(
            files=split['files_train'],
            date_A=split['buffer_st'],
            date_B=split['buffer_end'],
            range_type='exclude',
            n_historical=n_historical,
            p_mask=p_mask,
        ),
        'valid': FinancialStatementsDataset(
            files=split['files_valid'],
            date_A=split['valid_st'],
            date_B=split['valid_end'],
            range_type='include',
            n_historical=n_historical,
            p_mask=p_mask,
        ),
    })

    print(f'''{i:<3} T: {len(datasets[-1]['train']):>8,}  V: {len(datasets[-1]['valid']):>8,}''')


# # Model

# +
class EncoderBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        
        self.reducer = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=input_size),
            nn.Linear(input_size, output_size),
        )
        
        self.sequencer = nn.GRU(
            input_size=output_size,
            hidden_size=output_size//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, seq_len, n_features)
        """
        batch_size, seq_len, n_features = x.shape
        x = x.reshape(-1, n_features)
        x = self.reducer(x)
        x = x.reshape(batch_size, seq_len, -1)
        x, _ = self.sequencer(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super().__init__()
        
        self.sequencer = nn.GRU(
            input_size=input_size,
            hidden_size=input_size//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        self.expander = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=input_size),
            nn.Linear(input_size, output_size),
        )
        
    def forward(self, x):
        """
        Args:
            x (tensor): (batch_size, seq_len, n_features)
        """
        x, _ = self.sequencer(x)
        batch_size, seq_len, n_features = x.shape
        x = x.reshape(-1, n_features)
        x = self.expander(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x

    
class StatementEncoder(nn.Module):
    """U-net encoder for n-quarters of financial statements
    """
    def __init__(self, 
                 input_size,  
                 seq_len=20, 
                 block_sizes=[256, 128, 128, 64, 64], 
                 layer_dropout=0.1, 
                 cross_connection_dropout=0.5): 
        
        super().__init__()

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(n_in, n_out, layer_dropout)
            for n_in, n_out
            in zip([input_size] + block_sizes[:-1], block_sizes)
        ])
        
        # Encoder final reducer
        self.reducer_fxn = lambda x: torch.logsumexp(x, dim=1)
        self.reducer_norm = nn.LayerNorm(block_sizes[-1])

        # Decoder initial expander
        self.expander = nn.Linear(block_sizes[-1],  block_sizes[-1]*seq_len)
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(n_in, n_out, layer_dropout)
            for n_in, n_out
            in zip(block_sizes[::-1], block_sizes[:-1][::-1] + [input_size])
        ])

        # Dropout for cross connection
        self.cc_dropout = nn.Dropout(p=cross_connection_dropout)

    def encode(self, x):
        """
        
        Args:
            x (tensor): (batch_size, seq_len, n_statement_features)
        """
        for block in self.encoder_blocks:
            x = block(x)
            
        # Flatten output
        x = self.reducer_fxn(x)
        x = self.reducer_norm(x)
        
        return x

    
    def forward(self, x):
        """
        
        Args:
            x (tensor): (batch_size, seq_len, n_statement_features)
        """
        batch_size, seq_len, _ = x.shape
        
        ## Encoding ###
        encoder_states = []
        for block in self.encoder_blocks:
            x = block(x)
            encoder_states.append(x)
            
        # Flatten output
        x = self.reducer_fxn(x)
        x = self.reducer_norm(x)
        
        ### Decoding ###
        # Expand output
        x = self.expander(x).reshape(batch_size, seq_len, -1)
        
        encoder_states = encoder_states[:-1][::-1] ## Slice off last state and reverse
        for block, encoder_state in zip(self.decoder_blocks[:-1], encoder_states):
            x = block(x) + self.cc_dropout(encoder_state)
        x = self.decoder_blocks[-1](x)
        
        return x
# -

# # Train model

# +
model = StatementEncoder(input_size=267)
optimizer = torch.optim.Adam(params=model.parameters())
def loss_fxn(yh, y, non_zero_wt=100.0):
    wts = torch.where(y == 0, 1.0, non_zero_wt)
    loss = nn.L1Loss(reduction='none')(yh, y)*wts
    return torch.sum(loss)/torch.sum(wts)

dataset = datasets[13]


model.eval()
loader = DataLoader(dataset['valid'], batch_size=1, shuffle=False)
valid_losses = []
for batch in loader:
    x, y = batch.values()

    with torch.no_grad():
        yh = model(x)
        loss = loss_fxn(yh, y)


    valid_losses.append(loss.item())
print(f'{np.mean(valid_losses):>6.3}')

# +
idx = np.argmax(valid_losses)
x, y = dataset['valid'][idx].values()

model.eval()
with torch.no_grad():
    e = model.encode(x.unsqueeze(0))
    yh = model(x.unsqueeze(0))

print(e.squeeze().numpy())

fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(x.numpy(), cmap='RdBu', vmin=-2, vmax=2)

fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(y.numpy(), cmap='RdBu', vmin=-2, vmax=2)

fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(yh.squeeze(0).numpy(), cmap='RdBu', vmin=-2, vmax=2)

# +
n_epochs = 100
lr_lower, lr_upper = 0.00005, 0.005
batch_size = 128
n_batches = int(len(dataset['train'])/batch_size) + 1

for i in range(n_epochs):
    model.train()
    loader = DataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
    train_losses = []
    for j, batch in enumerate(loader, 1):
        # Adjust learning rate 
        lr = lr_lower + (np.sin((j - n_batches//4)/(n_batches//2)*np.pi) + 1)/2*(lr_upper - lr_lower)
        for parm_group in optimizer.param_groups:
            parm_group['lr'] = lr
            
        # Forward pass
        x, y = batch.values()
        yh = model(x)
        loss = loss_fxn(yh, y)
        train_losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Print progress
        if j%(n_batches//20) == 0:
            print(f'{j/n_batches:.0%}', end=' ')
        
    model.eval()
    loader = DataLoader(dataset['valid'], batch_size=1, shuffle=False)
    valid_losses = []
    for batch in loader:
        x, y = batch.values()
        
        with torch.no_grad():
            yh = model(x)
            loss = loss_fxn(yh, y)
            
        valid_losses.append(loss.item())
    
    print()
    t_mean = np.mean(train_losses)
    t_min = np.min(train_losses)
    t_max = np.max(train_losses)
    t_p = np.mean(train_losses < t_mean)
    v_mean = np.mean(valid_losses)
    print(f'{i:>2}, {t_mean:>6.4} ({t_min:>6.4} - {t_max:>6.4}; {t_p:>4.0%}), {v_mean:>6.4}')

# +
x, y = dataset['valid'][idx].values()

model.eval()
with torch.no_grad():
    e = model.encode(x.unsqueeze(0))
    yh = model(x.unsqueeze(0))

print(e.squeeze().numpy())


fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(x.numpy(), cmap='RdBu', vmin=-3, vmax=3)

fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(y.numpy(), cmap='RdBu', vmin=-3, vmax=3)

fig, ax = plt.subplots(figsize=(20,7))
_ = ax.imshow(yh.squeeze(0).numpy(), cmap='RdBu', vmin=-3, vmax=3)
# -

valid_losses

# # Encoding clusters

# +
model.eval()
loader = DataLoader(dataset['train'], batch_size=64, shuffle=True)

encodings = []
for batch in loader:
    x, y = batch.values()

    with torch.no_grad():
        encodings.append(model.encode(x).numpy())
        
encodings = np.concatenate(encodings, axis=0)
encodings.shape

# +
import umap

reducer = umap.UMAP(
    n_components=2,
    random_state=1234
)
embedding = reducer.fit_transform(encodings)
# -

fig, ax = plt.subplots(figsize=(10,10))
_ = ax.scatter(*embedding.T, alpha=0.1)


