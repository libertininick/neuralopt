# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: neuralopt_env
#     language: python
#     name: neuralopt_env
# ---

# +
import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# Plotting style
plt.style.use('seaborn-colorblind')
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = 'white'

DATA_PATH = '../../../Investing Models/Price data'

# +
df = pd.read_csv(f'{DATA_PATH}/PCG.csv')
df['return'] = df['Adj Close'].pct_change(periods=1)
df['high_to_close'] = df['High']/df['Close'] - 1
df['low_to_close'] = df['Low']/df['Close'] - 1

# Rolling 120d return
df['return_120d'] = df['return'].rolling(120).mean()

# Average true range
df['prev_close'] = df['Adj Close'].shift(periods=1)
df['true_range'] = np.maximum(df['prev_close'], df['High'])/np.minimum(df['prev_close'], df['Low']) - 1
df['true_range_pct'] = df['true_range'].rolling(20).apply(lambda x: (x.tail(1) - np.min(x))/(np.max(x) - np.min(x)))
df['atr_20d'] = df['true_range'].rolling(20).mean()
df['atr_3yr_pct'] = df['atr_20d'].rolling(250*3).apply(lambda x: (x.tail(1) - np.min(x))/(np.max(x) - np.min(x)))

# Normalize returns based on rolling 20-day average true range
df['return_norm'] = df['return']/df['atr_20d'].shift(periods=1)
df['high_to_close_norm'] = df['high_to_close']/df['atr_20d'].shift(periods=1)
df['low_to_close_norm'] = df['low_to_close']/df['atr_20d'].shift(periods=1)

# Forward looking returns
for step in range(1, 41):
    df[f'return_t{step}'] = ((df['Adj Close'].shift(periods=-step)/df['Adj Close'] - 1) - df['return_120d']*step)/(df['atr_20d']*(step**0.5))

# Drop NA rows
df = df.dropna(how='any', axis='rows')
# -



bins = [7.5/(1.5**i) for i in range(15)]
bins = np.array([-b for b in bins] + [0] + bins[::-1])
# bins = np.quantile(df['return_t10'], q=np.linspace(0,1,100))
bin_threshs = np.array([np.mean(df['return_t10'] <= b) for b in bins])
for i, b in enumerate(bins):
    print(f'''{i:<3} {b:>8.4f}: {np.mean(df['return_t10'] <= b):>10.4%}''')

x = np.random.rand(10000)
u_idx = np.searchsorted(bin_threshs, x, side='left', sorter=None)
l_idx = u_idx - 1
u_thresh = bin_threshs[u_idx]
l_thresh = bin_threshs[l_idx]
u_value = bins[u_idx]
l_value = bins[l_idx]
interp_pct = (x - l_thresh)/(u_thresh - l_thresh)
v = (u_value - l_value)*interp_pct + l_value
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(v, bins=100)

fig, ax = plt.subplots(figsize=(10,5))
q =  np.quantile(df['return_t10'], q=np.linspace(0.01,0.99,40))
pt = [min(q), max(q)]
_ = ax.plot(pt, pt, "k:")
_ = ax.scatter(
    q,
    np.quantile(v, q=np.linspace(0.01,0.99,40))
)

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_t1'], bins=100)
_ = ax.hist(df['return_t2'], bins=100)
_ = ax.hist(df['return_t5'], bins=100)
_ = ax.hist(df['return_t10'], bins=100)

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_norm'], bins=100)


