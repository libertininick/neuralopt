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

DATA_PATH = '../../../Investing Models'
# -

# # Load earnings data

df_earnings = pd.read_csv(f'{DATA_PATH}/Earnings data/all_earnings_data.csv')
df_earnings.info()

# # Data transform

# +
from scipy.spatial.distance import cdist

def calc_earnings_dists(dates, earnings_dates):
    """Calculates the number of calendar days since the last earnings release
    and the number of days until the next earnings for each date in `dates`
    
    Args:
        dates (ndarray): Sequence of trading dates
        earnings_dates (ndarray): Earnings release dates
    """
    
    days = cdist(
        dates[:, None], 
        earnings_dates[:, None], 
        lambda a,b: (b - a)/np.timedelta64(1, 'D')
    )
    
    days_since = np.abs(np.max(np.where(days <= 0, days, -np.inf), axis=-1))
    days_until = np.min(np.where(days >= 0, days, np.inf), axis=-1)
    
    return days_since, days_until


def transform(df, n_forward=40):
    """Transforms raw price data to model inputs
    
    Args:
        df (DataFrame): Raw price inputs
        n_forward (int): Number of forward looking returns
        
    Returns:
        df (DataFrame): Transformed inputs
    """
    
    # Calculate adjustment factor
    df['adj_factor'] = df['Adj Close']/df['Close']
    df['Adj High'] = df['adj_factor']*df['High']
    df['Adj Low'] = df['adj_factor']*df['Low']

    # Raw returns
    df['return'] = df['Adj Close'].pct_change(periods=1)
    df['high_to_close'] = df['Adj High']/df['Adj Close'] - 1
    df['low_to_close'] = df['Adj Low']/df['Adj Close'] - 1

    # Rolling 120d returns
    df['return_120d'] = df['return'].ewm(span=120).mean()
    df['high_to_close_120d'] = df['high_to_close'].ewm(span=120).mean()
    df['low_to_close_120d'] = df['low_to_close'].ewm(span=120).mean()

    # Average true range
    df['prev_close'] = df['Adj Close'].shift(periods=1)
    df['true_range'] = np.maximum(df['prev_close'], df['Adj High'])/np.minimum(df['prev_close'], df['Adj Low']) - 1
    df['atr_20d'] = df['true_range'].ewm(span=20).mean()
    df['stdtr_20d'] = df['true_range'].ewm(span=20).std()
    df['true_range_norm'] = (df['true_range'] - df['atr_20d'])/df['stdtr_20d']
    df['atr_3yr_pct'] = df['atr_20d'].rolling(250*3).apply(lambda x: (x.tail(1) - np.min(x))/(np.max(x) - np.min(x)))

    # Normalize returns based on rolling 120-day mean and rolling 20-day average true range
    df['return_norm'] = (df['return'] - df['return_120d'])/df['atr_20d']
    df['high_to_close_norm'] = (df['high_to_close'] - df['high_to_close_120d'])/df['atr_20d']
    df['low_to_close_norm'] = (df['low_to_close'] - df['low_to_close_120d'])/df['atr_20d']

    # Forward looking returns
    for step in range(1, n_forward + 1):
        forward_return = df['Adj Close'].shift(periods=-step)/df['Adj Close'] - 1
        scaled_mean = df['return_120d']*step
        scaled_dev = df['atr_20d']*step**0.5
        forward_norm = (forward_return - scaled_mean)/scaled_dev
        df[f'return_t{step}'] = forward_norm
            
    # Drop NA rows
    df = df.dropna(how='any', axis='rows')

    # Select columns
    cols = [
        'Date',
        'Symbol',
        'return_120d',
        'atr_20d',
        'return_norm',
        'high_to_close_norm',
        'low_to_close_norm',
        'true_range_norm',
        'atr_3yr_pct',
    ]
    cols += [f'return_t{step}' for step in range(1, n_forward + 1)]
    df = df.loc[:, cols]
    
    return df


# -

ticker = 'MSFT'
df = pd.read_csv(f'{DATA_PATH}/Price data/{ticker}.csv')
dates = pd.to_datetime(df['Date']).values
earnings_dates = pd.to_datetime(df_earnings.query('''Symbol == @ticker''')['Date']).values

days_since, days_until = calc_earnings_dists(dates, earnings_dates)
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.plot(days_since)

# +

df = transform(df)

# +
fig, ax = plt.subplots(figsize=(15,10))

_ = ax.plot(df['atr_20d'])

# +
fig, ax = plt.subplots(figsize=(15,10))
_ = ax.hist(df['return_t1'], bins=100, alpha=0.5, label='t1')
_ = ax.hist(df['return_t2'], bins=100, alpha=0.5, label='t2')
_ = ax.hist(df['return_t5'], bins=100, alpha=0.5, label='t5')
_ = ax.hist(df['return_t10'], bins=100, alpha=0.5, label='t10')
_ = ax.hist(df['return_t20'], bins=100, alpha=0.5, label='t20')
_ = ax.hist(df['return_t40'], bins=100, alpha=0.5, label='t40')

_ = ax.legend()
# -

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['true_range_norm'], bins=100)

np.mean(df['return_norm']), np.std(df['return_norm'])


# # Sample from bins function

def sample_from_bins(n, bins, bin_ppfs):
    """Random sample using linear interpolation between bins based on bin ppfs
    
    Args:
        n (int): Number of samples
        bins (ndarray): Bin thresholds (<=)
        bin_ppfs (ndarray): Emperical or predicted ppf for each bin
    """
    
    # Sample uniform random numbers
    x = np.random.rand(n)
    
    # Find upper and lower bin indexes for each random draw based on ppfs
    u_idx = np.searchsorted(bin_ppfs, x, side='left', sorter=None)
    l_idx = u_idx - 1
    
    # PPFs
    u_ppf = bin_ppfs[u_idx]
    l_ppf = bin_ppfs[l_idx]
    
    # Values based on linear interpolation between upper and lower bins
    u_bin = bins[u_idx]
    l_bin = bins[l_idx]
    interp_pct = (x - l_ppf)/(u_ppf - l_ppf)
    values = (u_bin - l_bin)*interp_pct + l_bin
    
    return values


# +
from scipy.stats import norm

bins = [-10, -7.5, -5] + [norm.ppf(i) for i in np.linspace(0.001, 0.499, 22)]
bins += [0] + [-b for b in bins[::-1]]
bins = np.array(bins)
bin_ppfs = np.array([np.mean(df['return_t10'] <= b) for b in bins])
# -



v = sample_from_bins(len(df['return_t10']), bins, bin_ppfs)
fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_t10'], bins=100, alpha=0.5)
_ = ax.hist(v, bins=100, alpha=0.5, color='orange')

fig, ax = plt.subplots(figsize=(10,5))
q =  np.quantile(df['return_t10'], q=np.linspace(0.01,0.99,40))
pt = [min(q), max(q)]
_ = ax.plot(pt, pt, "k:")
_ = ax.scatter(
    q,
    np.quantile(v, q=np.linspace(0.01,0.99,40))
)

fig, ax = plt.subplots(figsize=(10,5))
_ = ax.hist(df['return_norm'], bins=100)


