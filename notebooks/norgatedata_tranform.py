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

import os
import sys

import norgatedata
import pandas as pd

# Norgate settings
priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
padding_setting = norgatedata.PaddingType.ALLMARKETDAYS
timeseriesformat = 'pandas-dataframe'

# Add path to repo
wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)
p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)

from data_utils import calc_earnings_markers, transform_prices
# -
# # Load, save, and transform data


start_date = '1990-01-01'
watchlistname = 'Russell 1000 Current & Past'
symbols = norgatedata.watchlist_symbols(watchlistname)
print(f'# Symbols: {len(symbols):,}')

earnings_dates = pd.read_csv('../../../Investing Models/Data/Earnings/all_earnings_data.csv')
earnings_dates['Date'] = pd.to_datetime(earnings_dates['Date'], format='%Y-%m-%d', errors='coerce')
earnings_dates.shape

for symbol in symbols:
    prices = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting = priceadjust,
        padding_setting = padding_setting,
        start_date = start_date,
        timeseriesformat=timeseriesformat,
    )

    prices = norgatedata.index_constituent_timeseries(
        symbol,
        'S&P 500',
        padding_setting = padding_setting,
        limit = -1,
        pandas_dataframe = prices,
        timeseriesformat = timeseriesformat,
    ).rename({'Index Constituent': 'S&P 500 Constituent'}, axis='columns')

    prices = norgatedata.index_constituent_timeseries(
        symbol,
        'NASDAQ 100',
        padding_setting = padding_setting,
        limit = -1,
        pandas_dataframe = prices,
        timeseriesformat = timeseriesformat,
    ).rename({'Index Constituent': 'NASDAQ 100 Constituent'}, axis='columns')
    
    prices = norgatedata.index_constituent_timeseries(
        symbol,
        'Russell 1000',
        padding_setting = padding_setting,
        limit = -1,
        pandas_dataframe = prices,
        timeseriesformat = timeseriesformat,
    ).rename({'Index Constituent': 'Russell 1000 Constituent'}, axis='columns')
    
    prices['earnings_marker'] = calc_earnings_markers(
        trading_dates=prices.index, 
        earnings_dates=earnings_dates.query('''Symbol == @symbol''').Date
    )
    
    prices.to_csv(f'../../../Investing Models/Data/R1K member daily prices/{symbol}.csv')
    
    r1k_mask = prices['Russell 1000 Constituent'] == 1
    prices = prices[r1k_mask]
    if len(prices) > 300:
        transform_prices(prices).to_pickle(f'D:/opt/Price Transforms/{symbol}.pkl')

# ## Add earnings markets from CSV price data

# +
from glob import glob
import re

files = glob('../../../Investing Models/Data/R1K member daily prices/*.csv')
p = re.compile(r'[^\\\\|\/]{1,100}(?=\.csv$)')
symbols = [p.findall(file)[0] for file in files]

for symbol, file in zip(symbols, files):
    prices = pd.read_csv(f'../../../Investing Models/Data/R1K member daily prices/{symbol}.csv')
    prices['Date'] = pd.to_datetime(prices['Date'], format='%Y-%m-%d', errors='coerce')
    prices = prices.set_index('Date', drop=True)
    
    prices['earnings_marker'] = calc_earnings_markers(
        trading_dates=prices.index, 
        earnings_dates=earnings_dates.query('''Symbol == @symbol''').Date
    )
    
    r1k_mask = prices['Russell 1000 Constituent'] == 1
    prices = prices[r1k_mask]
    if len(prices) > 300:
        transform_prices(prices).to_pickle(f'D:/opt/Price Transforms/{symbol}.pkl')
# -


