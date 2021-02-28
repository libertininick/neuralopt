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
import random
import sys

import pandas as pd
import norgatedata
import numpy as np

priceadjust = norgatedata.StockPriceAdjustmentType.TOTALRETURN
padding_setting = norgatedata.PaddingType.ALLMARKETDAYS
timeseriesformat = 'pandas-dataframe'

wd = os.path.abspath(os.getcwd())
path_parts = wd.split(os.sep)
p = path_parts[0]
for part in path_parts[1:]:
    p = p + os.sep + part
    if p not in sys.path:
        sys.path.append(p)

from data_utils import transform_prices
# -



start_date = '1990-01-01'
watchlistname = 'NASDAQ 100 Current & Past'
symbols = norgatedata.watchlist_symbols(watchlistname)



# +
symbol = symbols[1]
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
# -

3.825353e+09/89325728.0

prices.info()

t = transform_prices(prices)

np.quantile(t['turnover_norm'], np.linspace(0,1,11))

t['CC_norm'].rolling(20).sum()

import torch
import torch.nn as nn

x = torch.randn((1,64,4*256,1))
x.shape

# +
m = nn.Conv2d(
    in_channels=64, 
    out_channels=64, 
    kernel_size=(2,1), 
    stride=2
)

m2 = nn.ConvTranspose2d(
    in_channels=64, 
    out_channels=64, 
    kernel_size=(2,1), 
    stride=2
)
# -

with torch.no_grad():
    y = m(x)
    x2 = m2(y)

y.shape

x2.shape

file ='../../../Investing Models/Data/Prices/APPL.pkl'

import re

re.match(pattern=r'(?<=\//),(?=\d)')
