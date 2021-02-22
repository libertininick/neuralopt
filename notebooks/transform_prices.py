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
#     display_name: neuralopt_env
#     language: python
#     name: neuralopt_env
# ---

# +
# %load_ext autoreload
# %autoreload 2

import calendar
import glob
import os
import random
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

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

from data_utils import transform
# -

# # Load rates data

df_rates = pd.read_csv(f'{DATA_PATH}/Rates/daily_treasury_yield_curve_rates.csv')
df_rates['risk_free_rate'] = df_rates['3 Mo']/100
df_rates = df_rates[['Date', 'risk_free_rate']]
df_rates['Date'] = pd.to_datetime(df_rates['Date'])

# # Load earnings data

df_earnings = pd.read_csv(f'{DATA_PATH}/Earnings/all_earnings_data.csv')
df_earnings.info()

# # Transform and save

files = glob.glob(f'{DATA_PATH}/Prices/*.csv')
files = random.sample(files, 3000)

file = '../../../Investing Models/Data/Prices/AAPL.csv'
df_prices = pd.read_csv(file)
ticker = df_prices['Symbol'][0]
transform(
            df_prices=df_prices,
            df_rates=df_rates,
            earnings_dates=pd.to_datetime(df_earnings.query('''Symbol == @ticker''')['Date']).values,
        ).to_csv('test.csv')

# +
ct = 0
for i, file in enumerate(files, 1):
    # Load price data
    df_prices = pd.read_csv(file)
    ticker = df_prices['Symbol'][0]
    
    # Filter
    if (
        len(df_prices) >= 850 and 
        np.mean(df_prices['Adj Close'] == df_prices['Adj Close'].shift(1)) < 0.4 and
        np.all(df_prices['Adj Close'] > 0)
    ):

        # Transform
        df = transform(
            df_prices=df_prices,
            df_rates=df_rates,
            earnings_dates=pd.to_datetime(df_earnings.query('''Symbol == @ticker''')['Date']).values
        )
        
        # Check rolling returns == 0
        z = df['return'] == 0
        z = z.rolling(260).mean()
        if z.max() > 0.5:
            slice_idx = np.min(np.arange(len(z))[z > 0.5])
            df = df.iloc[:slice_idx]
            
        # Save
        if len(df) > 500 and (df['Date'] - df['Date'].shift(1)).dt.days.max() <= 10:
            ct += 1
            df.to_pickle(f'{DATA_PATH}/Transforms/{ticker}.pkl')
            
    print(i, ct)
        

#     if i%(len(files)//1000) == 0:
#         print(f'{i/len(files):.2%}')

# +
files = glob.glob(f'{DATA_PATH}/Transforms/*.pkl')
rets_norm = []
for i, file in enumerate(files):
    df = pd.read_pickle(file)
    rets_norm.append(df['return_norm'].values)
    gaps = (df['Date'] - df['Date'].shift(1)).dt.days
    
    print(f'''{i:<5} {int(np.max(gaps)):>3} {np.mean(df['return'] == 0):>6.2%} {np.median(df['return_norm']):>10.3f} {np.mean(df['return_norm']):>10.3f} {np.std(df['return_norm']):>10.3f}''')
    
    
rets_norm = np.concatenate(rets_norm)
# -

df = pd.read_pickle(files[1082])
df.to_csv('test.csv')

fig, ax = plt.subplots(figsize=(15,7))
mask = np.logical_and(rets_norm >= -10, rets_norm <= -2.5)
_ = ax.hist(rets_norm[mask], bins=500)


# ## Viz transforms



fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_norm'], bins=100, edgecolor='black')

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['true_range_norm'], bins=100, edgecolor='black')

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['tr_avg_norm'], bins=100, edgecolor='black')

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['choppiness_260d'], bins=100, edgecolor='black')

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['cdf_x2'], bins=100, edgecolor='black')

# +
fig, ax = plt.subplots(figsize=(15,10))

_ = ax.plot(df['cdf_I'])
# -

# ## Bin sampling of normalized returns

# +
from scipy.stats import norm

from pricing_utils import sample_from_bins

df = pd.read_pickle(f'{DATA_PATH}/Transforms/A.pkl')

bins = [-10, -7.5, -5] + [norm.ppf(i) for i in np.linspace(0.001, 0.499, 22)]
bins += [0] + [-b for b in bins[::-1]]
bins = np.array(bins)
bin_ppfs = np.array([np.mean(df['return_norm'] <= b) for b in bins])

v = sample_from_bins(len(df['return_norm']), bins, bin_ppfs)
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_norm'], bins=100, alpha=0.5)
_ = ax.hist(v, bins=100, alpha=0.5, color='orange')
# -


