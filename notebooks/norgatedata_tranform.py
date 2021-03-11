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

from data_utils import transform_prices
# -
# # Load, save, and transform data


start_date = '1990-01-01'
watchlistname = 'Russell 1000 Current & Past'
symbols = norgatedata.watchlist_symbols(watchlistname)
print(f'# Symbols: {len(symbols):,}')

# +
for symbol in symbols[100:110]:
    prices = norgatedata.price_timeseries(
        symbol,
        stock_price_adjustment_setting = priceadjust,
        padding_setting = padding_setting,
        start_date = start_date,
        timeseriesformat=timeseriesformat,
    )

#     prices = norgatedata.index_constituent_timeseries(
#         symbol,
#         'S&P 500',
#         padding_setting = padding_setting,
#         limit = -1,
#         pandas_dataframe = prices,
#         timeseriesformat = timeseriesformat,
#     ).rename({'Index Constituent': 'S&P 500 Constituent'}, axis='columns')

#     prices = norgatedata.index_constituent_timeseries(
#         symbol,
#         'NASDAQ 100',
#         padding_setting = padding_setting,
#         limit = -1,
#         pandas_dataframe = prices,
#         timeseriesformat = timeseriesformat,
#     ).rename({'Index Constituent': 'NASDAQ 100 Constituent'}, axis='columns')
    
    prices = norgatedata.index_constituent_timeseries(
        symbol,
        'Russell 1000',
        padding_setting = padding_setting,
        limit = -1,
        pandas_dataframe = prices,
        timeseriesformat = timeseriesformat,
    ).rename({'Index Constituent': 'Russell 1000 Constituent'}, axis='columns')
    
    r1k_mask = prices['Russell 1000 Constituent'] == 1
    prices = prices[r1k_mask]
    if len(prices) > 300:
        transform_prices(prices).to_pickle(f'../../../Investing Models/Data/Price Transforms/{symbol}.pkl')
